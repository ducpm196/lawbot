"""
Datasets Module - LawBot v8.0
=============================

Dataset classes for LawBot including:
- Bi-encoder training datasets
- Reranker training datasets
- Data loading and processing utilities
"""

from .legal_qa import BiEncoderDataset, RerankerDataset

__all__ = ["BiEncoderDataset", "RerankerDataset"]
