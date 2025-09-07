from typing import Any, Dict, List, Optional, Union
import re

import unicodedata

from core.utils.canonicalization import canonicalize_aid

"""
Base Transforms - LawBot v8.0
=============================

Base classes and utility transforms for data processing.
"""


class BaseTransform:
    """Base class for all transforms."""

    def __call__(self, data: Any) -> Any:
        """Apply the transform to the data.

        Args:
            data: Input data to transform

        Returns:
            Transformed data

        Raises:
            NotImplementedError: If the transform is not implemented
        """
        raise NotImplementedError(
            "Each transform must implement the `__call__` method."
        )


class Compose(BaseTransform):
    """Composes several transforms together."""

    def __init__(self, transforms: List[BaseTransform]):
        """Initialize the composition of transforms.

        Args:
            transforms: List of transforms to compose
        """
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        """Apply all transforms in sequence.

        Args:
            data: Input data to transform

        Returns:
            Data after applying all transforms
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def add_transform(self, transform: BaseTransform):
        """Add a new transform to the composition.

        Args:
            transform: Transform to add
        """
        self.transforms.append(transform)

    def get_transform_count(self) -> int:
        """Get the number of transforms in the composition.

        Returns:
            Number of transforms
        """
        return len(self.transforms)


class TextProcessor(BaseTransform):
    """A unified text processor for cleaning, normalizing, and transforming text."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_extra_spaces: bool = True,
    ):
        """Initializes the text processor.

        Args:
            normalize_unicode: Whether to apply Unicode normalization (NFKC).
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation.
            remove_extra_spaces: Whether to collapse multiple whitespace characters.
        """
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_spaces = remove_extra_spaces

    def __call__(self, text: str) -> str:
        """Applies the configured text transformations.

        Args:
            text: The input string to process.

        Returns:
            The processed string.
        """
        if not isinstance(text, str):
            return ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            # This regex keeps word characters and spaces
            text = re.sub(r"[^\w\s]", "", text)

        # Replace newlines and tabs with a single space
        text = re.sub(r"[\n\t]", " ", text)

        if self.remove_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()

        return text


class LegalTextCleaner(TextProcessor):
    """A specialized text processor for cleaning Vietnamese legal documents."""

    def __init__(self):
        """Initializes the legal text cleaner with default settings for legal text."""
        super().__init__(
            normalize_unicode=True,
            lowercase=False,  # Legal text often has meaningful capitalization
            remove_punctuation=False,  # Punctuation can be important in legal text
            remove_extra_spaces=True,
        )

        # Pre-compiled regex for efficiency
        self.article_pattern = re.compile(
            r"(Điều|Khoản|Điểm)\s+\d+[a-z]?\.?", re.IGNORECASE
        )
        self.abbreviation_map = {
            "QH": "Quốc hội",
            "UBTVQH": "Ủy ban Thường vụ Quốc hội",
            "CP": "Chính phủ",
            "NĐ-CP": "Nghị định - Chính phủ",
            "TTg": "Thủ tướng Chính phủ",
            "BTC": "Bộ Tài chính",
            "BTP": "Bộ Tư pháp",
            "TANDTC": "Tòa án nhân dân tối cao",
            "VKSNDTC": "Viện kiểm sát nhân dân tối cao",
        }

    def __call__(self, text: str) -> str:
        """Applies base processing and then legal-specific cleaning rules.

        Args:
            text: The legal text to clean.

        Returns:
            The cleaned legal text.
        """
        # Apply base processing first
        text = super().__call__(text)

        # Remove article/clause/point numbers (e.g., "Điều 1.", "Khoản 2.")
        text = self.article_pattern.sub(r"\1", text)

        # Replace common abbreviations
        # Use a regex to avoid replacing parts of words
        for abbr, full_text in self.abbreviation_map.items():
            text = re.sub(rf"\b{abbr}\b", full_text, text)

        # Final cleanup of extra spaces that might have been introduced
        text = re.sub(r"\s+", " ", text).strip()

        return text
