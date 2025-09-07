from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from core.utils.logging_manager import get_logger

"""
Legal QA Datasets - LawBot v8.1 (Refactored)
============================================

Refactored, efficient, and self-contained Dataset classes for training
bi-encoder and reranker models for legal question answering.
"""

logger = get_logger(__name__)


class BiEncoderDataset(Dataset):
    """Dataset for training sentence-transformer-style bi-encoder models."""

    def __init__(
    self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, max_length: int):
        """Initializes the bi-encoder dataset.

        Args:
            data: A list of dictionaries, each with 'query', 'positive', and 'negative' keys.
            tokenizer: The tokenizer for text processing.
            max_length: The maximum sequence length for tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(
            f"BiEncoderDataset initialized with {len(data)} examples, max_length={max_length}")

    def __len__(self) -> int:
        """__len__ function."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """__getitem__ function."""
        item = self.data[idx]
        query = item.get("query", "")
        positive = item.get("positive", "")
        negative = item.get("negative", "")

        # Tokenize each part separately for bi-encoder training
        query_inputs = self.tokenizer(
    query, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        positive_inputs = self.tokenizer(
    positive, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        negative_inputs = self.tokenizer(
    negative, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            # Squeeze to remove the batch dimension of 1
            "query_input_ids": query_inputs["input_ids"].squeeze(0),
            "query_attention_mask": query_inputs["attention_mask"].squeeze(0),
            "positive_input_ids": positive_inputs["input_ids"].squeeze(0),
            "positive_attention_mask": positive_inputs["attention_mask"].squeeze(0),
            "negative_input_ids": negative_inputs["input_ids"].squeeze(0),
            "negative_attention_mask": negative_inputs["attention_mask"].squeeze(0),
        }


class RerankerDataset(Dataset):
    """Dataset for training cross-encoder reranker models."""

    def __init__(
    self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_length: int):
        """Initializes the reranker dataset.

        Args:
            data: A list of dictionaries, each with 'query', 'passage', and 'label' keys.
            tokenizer: The tokenizer for text processing.
            max_length: The maximum sequence length for tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(
            f"RerankerDataset initialized with {len(data)} examples, max_length={max_length}")

    def __len__(self) -> int:
        """__len__ function."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """__getitem__ function."""
        item = self.data[idx]
        query = item.get("query", "")
        passage = item.get("passage", "")
        label = item.get("label", 0)

        # Tokenize the query-passage pair for cross-encoder training
        inputs = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            # Squeeze to remove the batch dimension of 1
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }
