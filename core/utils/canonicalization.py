import re
from typing import Union

"""
Canonicalization Utilities - LawBot v8.0
========================================

Utility functions for standardizing and normalizing legal document identifiers.
"""

def canonicalize_aid(law_id: str, article_id: Union[str, int]) -> str:
    """Creates a standardized Article ID.

    Args:
        law_id: The law identifier (e.g., "123/2020/QH14")
        article_id: The article identifier (e.g., "15" or "15a")

    Returns:
        Canonicalized AID string (e.g., "1232020QH14_15A")

    Examples:
        >>> canonicalize_aid("123/2020/QH14", "15")
        '1232020QH14_15'
        >>> canonicalize_aid("456-2021", "20a")
        '4562021_20A'
    """
    if not law_id or not article_id:
        raise ValueError("Both law_id and article_id must be provided")

    # Clean and standardize law_id
    law_id_cleaned = re.sub(r"[-\s_]", "", str(law_id)).upper()

    # Clean and standardize article_id
    article_id_cleaned = re.sub(r"[-\s_]", "", str(article_id)).upper()

    # Create canonical AID
    canonical_aid = f"{law_id_cleaned}_{article_id_cleaned}"

    return canonical_aid


def parse_aid(canonical_aid: str) -> tuple[str, str]:
    """Parse a canonical AID back into its components.

    Args:
        canonical_aid: Canonical AID string

    Returns:
        Tuple of (law_id, article_id)

    Examples:
        >>> parse_aid("1232020QH14_15")
        ('1232020QH14', '15')
    """
    if not canonical_aid or "_" not in canonical_aid:
        raise ValueError("Invalid canonical AID format")

    parts = canonical_aid.split("_", 1)
    if len(parts) != 2:
        raise ValueError("Invalid canonical AID format")

    law_id, article_id = parts
    return law_id, article_id


def is_valid_aid(canonical_aid: str) -> bool:
    """Check if a canonical AID is valid.

    Args:
        canonical_aid: Canonical AID string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_aid(canonical_aid)
        return True
    except ValueError:
        return False
