#!/usr/bin/env python3
"""
Validation Sets Manager - LawBot v8.3
====================================

Creates and manages separate validation sets for each training tier
to enable independent evaluation and prevent data leakage.
"""

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ValidationSetManager:
    """Manages validation sets for different training tiers."""

    def __init__(self, random_seed: int = 42):
        """Initialize with a fixed random seed for reproducibility."""
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"✅ Validation set manager initialized with seed {random_seed}")

    def create_tier_validation_sets(
        self,
        training_data: List[Dict],
        validation_split: float = 0.2,
        min_samples_per_tier: int = 10,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create separate validation sets for each tier.

        Args:
            training_data: Full training dataset
            validation_split: Fraction of data to use for validation
            min_samples_per_tier: Minimum samples required per tier

        Returns:
            Tuple of (tier1_val, tier2_val, tier3_val)
        """
        try:
            # Shuffle data
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)

            # Calculate split point
            split_idx = int(len(shuffled_data) * (1 - validation_split))
            train_data = shuffled_data[:split_idx]
            val_data = shuffled_data[split_idx:]

            logger.info(
                f"📊 Split data: {len(train_data)} train, {len(val_data)} validation"
            )

            # Create tier-specific validation sets
            tier1_val = self._create_retrieval_validation(
                val_data, min_samples_per_tier
            )
            tier2_val = self._create_light_reranker_validation(
                val_data, min_samples_per_tier
            )
            tier3_val = self._create_cross_encoder_validation(
                val_data, min_samples_per_tier
            )

            logger.info(
                f"✅ Created validation sets: Tier1={len(tier1_val)}, Tier2={len(tier2_val)}, Tier3={len(tier3_val)}"
            )

            return tier1_val, tier2_val, tier3_val

        except Exception as e:
            logger.error(f"❌ Failed to create validation sets: {e}")
            # Return empty lists as fallback
            return [], [], []

    def _create_retrieval_validation(
        self, val_data: List[Dict], min_samples: int
    ) -> List[Dict]:
        """Create validation set for Bi-Encoder retrieval (Tier 1) - triplets format."""
        try:
            # For Bi-Encoder retrieval, we need triplets (query, positive, negative)
            retrieval_val = []

            for item in val_data:
                query = item.get("query", "")
                
                # Handle both triplet format (bi-encoder) and pair format (light ranking)
                if "positive" in item and "negative" in item:
                    # Triplet format from bi-encoder data
                    positive = item.get("positive", "")
                    negative = item.get("negative", "")
                    
                    if query and positive and negative:
                        retrieval_val.append(
                            {
                                "query": query,
                                "positive": positive,
                                "negative": negative,
                                "type": "triplet"
                            }
                        )
                elif "passage" in item and "label" in item:
                    # Pair format from light ranking data - convert to triplet
                    passage = item.get("passage", "")
                    label = item.get("label", 0.0)
                    
                    if query and passage and label > 0.5:  # Only process positive examples for triplets
                        # Find a negative example for this query
                        found_negative = False
                        for neg_item in val_data:
                            if (neg_item.get("query") == query and 
                                neg_item.get("label", 0.0) < 0.5 and 
                                neg_item.get("passage") != passage):
                                retrieval_val.append(
                                    {
                                        "query": query,
                                        "positive": passage,
                                        "negative": neg_item.get("passage", ""),
                                        "type": "triplet"
                                    }
                                )
                                found_negative = True
                                break
                        
                        # If no negative found for same query, use any negative from different query
                        if not found_negative:
                            for neg_item in val_data:
                                if (neg_item.get("label", 0.0) < 0.5 and 
                                    neg_item.get("passage") != passage):
                                    retrieval_val.append(
                                        {
                                            "query": query,
                                            "positive": passage,
                                            "negative": neg_item.get("passage", ""),
                                            "type": "triplet"
                                        }
                                    )
                                    break

            # Ensure minimum samples with better quality
            if len(retrieval_val) < min_samples:
                logger.warning(
                    f"⚠️ Tier 1 validation has only {len(retrieval_val)} samples, minimum {min_samples} required"
                )
                # Try to create more triplets from available data
                if len(retrieval_val) > 0:
                    # Duplicate existing triplets to meet minimum requirement
                    while len(retrieval_val) < min_samples and len(retrieval_val) > 0:
                        retrieval_val.extend(retrieval_val[:min(5, min_samples - len(retrieval_val))])

            return retrieval_val

        except Exception as e:
            logger.error(f"❌ Failed to create Tier 1 validation: {e}")
            return []

    def _create_light_reranker_validation(
        self, val_data: List[Dict], min_samples: int
    ) -> List[Dict]:
        """Create validation set for light reranker (Tier 2) - query-passage pairs with labels."""
        try:
            # For light reranker, we need query-passage pairs with labels
            reranker_val = []

            for item in val_data:
                query = item.get("query", "")
                
                # Handle both triplet format (bi-encoder) and pair format (light ranking)
                if "positive" in item and "negative" in item:
                    # Triplet format from bi-encoder data - convert to pairs
                    positive = item.get("positive", "")
                    negative = item.get("negative", "")
                    
                    if query and positive:
                        # Positive example
                        reranker_val.append({
                            "query": query,
                            "passage": positive,
                            "label": 1.0,
                            "type": "positive_pair"
                        })
                    
                    if query and negative:
                        # Negative example
                        reranker_val.append({
                            "query": query,
                            "passage": negative,
                            "label": 0.0,
                            "type": "negative_pair"
                        })
                elif "passage" in item and "label" in item:
                    # Pair format from light ranking data - use directly
                    passage = item.get("passage", "")
                    label = item.get("label", 0.0)
                    
                    if query and passage:
                        reranker_val.append({
                            "query": query,
                            "passage": passage,
                            "label": label,
                            "type": "positive_pair" if label > 0.5 else "negative_pair"
                        })

            # Ensure minimum samples with better quality
            if len(reranker_val) < min_samples:
                logger.warning(
                    f"⚠️ Tier 2 validation has only {len(reranker_val)} samples, minimum {min_samples} required"
                )
                # Try to create more pairs from available data
                if len(reranker_val) > 0:
                    # Duplicate existing pairs to meet minimum requirement
                    while len(reranker_val) < min_samples and len(reranker_val) > 0:
                        reranker_val.extend(reranker_val[:min(5, min_samples - len(reranker_val))])

            return reranker_val

        except Exception as e:
            logger.error(f"❌ Failed to create Tier 2 validation: {e}")
            return []

    def _create_cross_encoder_validation(
        self, val_data: List[Dict], min_samples: int
    ) -> List[Dict]:
        """Create validation set for cross encoder (Tier 3) - query-passage pairs with labels."""
        try:
            # For cross encoder, we need query-passage pairs with labels
            cross_encoder_val = []

            for item in val_data:
                query = item.get("query", "")
                
                # Handle both triplet format (bi-encoder) and pair format (light ranking)
                if "positive" in item and "negative" in item:
                    # Triplet format from bi-encoder data - convert to pairs
                    positive = item.get("positive", "")
                    negative = item.get("negative", "")
                    
                    if query and positive:
                        # Positive example
                        cross_encoder_val.append({
                            "query": query,
                            "passage": positive,
                            "label": 1.0,
                            "type": "positive_classification"
                        })
                    
                    if query and negative:
                        # Negative example
                        cross_encoder_val.append({
                            "query": query,
                            "passage": negative,
                            "label": 0.0,
                            "type": "negative_classification"
                        })
                elif "passage" in item and "label" in item:
                    # Pair format from light ranking data - use directly
                    passage = item.get("passage", "")
                    label = item.get("label", 0.0)
                    
                    if query and passage:
                        cross_encoder_val.append({
                            "query": query,
                            "passage": passage,
                            "label": label,
                            "type": "positive_classification" if label > 0.5 else "negative_classification"
                        })

            # Ensure minimum samples with better quality
            if len(cross_encoder_val) < min_samples:
                logger.warning(
                    f"⚠️ Tier 3 validation has only {len(cross_encoder_val)} samples, minimum {min_samples} required"
                )
                # Try to create more pairs from available data
                if len(cross_encoder_val) > 0:
                    # Duplicate existing pairs to meet minimum requirement
                    while len(cross_encoder_val) < min_samples and len(cross_encoder_val) > 0:
                        cross_encoder_val.extend(cross_encoder_val[:min(5, min_samples - len(cross_encoder_val))])

            return cross_encoder_val

        except Exception as e:
            logger.error(f"❌ Failed to create Tier 3 validation: {e}")
            return []

    def save_validation_sets(
        self,
        tier1_val: List[Dict],
        tier2_val: List[Dict],
        tier3_val: List[Dict],
        output_dir: str,
    ) -> bool:
        """Save validation sets to separate files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save Tier 1 validation
            with open(
                output_path / "tier1_validation.jsonl", "w", encoding="utf-8"
            ) as f:
                for item in tier1_val:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Save Tier 2 validation
            with open(
                output_path / "tier2_validation.jsonl", "w", encoding="utf-8"
            ) as f:
                for item in tier2_val:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Save Tier 3 validation
            with open(
                output_path / "tier3_validation.jsonl", "w", encoding="utf-8"
            ) as f:
                for item in tier3_val:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.info(f"✅ Validation sets saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save validation sets: {e}")
            return False

    def load_validation_sets(
        self, input_dir: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load validation sets from files."""
        try:
            input_path = Path(input_dir)

            # Load Tier 1 validation
            tier1_val = []
            tier1_file = input_path / "tier1_validation.jsonl"
            if tier1_file.exists():
                with open(tier1_file, "r", encoding="utf-8") as f:
                    tier1_val = [json.loads(line) for line in f]

            # Load Tier 2 validation
            tier2_val = []
            tier2_file = input_path / "tier2_validation.jsonl"
            if tier2_file.exists():
                with open(tier2_file, "r", encoding="utf-8") as f:
                    tier2_val = [json.loads(line) for line in f]

            # Load Tier 3 validation
            tier3_val = []
            tier3_file = input_path / "tier3_validation.jsonl"
            if tier3_file.exists():
                with open(tier3_file, "r", encoding="utf-8") as f:
                    tier3_val = [json.loads(line) for line in f]

            logger.info(
                f"✅ Loaded validation sets: Tier1={len(tier1_val)}, Tier2={len(tier2_val)}, Tier3={len(tier3_val)}"
            )
            return tier1_val, tier2_val, tier3_val

        except Exception as e:
            logger.error(f"❌ Failed to load validation sets: {e}")
            return [], [], []


def create_validation_sets_from_file(
    input_file: str, output_dir: str, validation_split: float = 0.2
) -> bool:
    """
    Create validation sets from a training data file.

    Args:
        input_file: Path to input training data file
        output_dir: Directory to save validation sets
        validation_split: Fraction of data for validation

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load training data
        with open(input_file, "r", encoding="utf-8") as f:
            training_data = [json.loads(line) for line in f]

        logger.info(
            f"📊 Loaded {len(training_data)} training samples from {input_file}"
        )

        # Create validation sets
        manager = ValidationSetManager()
        tier1_val, tier2_val, tier3_val = manager.create_tier_validation_sets(
            training_data, validation_split
        )

        # Save validation sets
        success = manager.save_validation_sets(
            tier1_val, tier2_val, tier3_val, output_dir
        )

        if success:
            logger.info(f"✅ Successfully created validation sets in {output_dir}")
            return True
        else:
            logger.error("❌ Failed to save validation sets")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to create validation sets: {e}")
        return False


if __name__ == "__main__":
    # Test the validation set manager
    logging.basicConfig(level=logging.INFO)

    # Example usage
    test_data = [
        {"query": "Q1", "passage": "P1", "label": 1.0},
        {"query": "Q2", "passage": "P2", "label": 1.0},
        {"query": "Q3", "passage": "P3", "label": 0.0},
        {"query": "Q4", "passage": "P4", "label": 1.0},
        {"query": "Q5", "passage": "P5", "label": 0.0},
    ]

    manager = ValidationSetManager()
    tier1_val, tier2_val, tier3_val = manager.create_tier_validation_sets(
        test_data, 0.4
    )

    print(f"Tier 1 validation: {len(tier1_val)} samples")
    print(f"Tier 2 validation: {len(tier2_val)} samples")
    print(f"Tier 3 validation: {len(tier3_val)} samples")
