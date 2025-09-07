import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import random

# Add project root to path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.loader import config
from core.transforms import LegalTextCleaner
from core.utils.canonicalization import canonicalize_aid
from core.utils.io import load_json, save_json, save_jsonl
from core.utils.logging_manager import setup_logging
from core.utils.versioning import generate_versioned_path, save_metadata

"""
Data Preparation Script - LawBot v8.0
====================================

Script for preparing training data from raw legal documents.
"""

# Setup workflow logging
setup_logging("workflow")
logger = logging.getLogger(__name__)


def normalize_data_format(data: List[Dict], target_format: str = "bi_encoder") -> List[Dict]:
    """Normalize data format for different model types.
    
    Args:
        data: List of training examples
        target_format: Target format ("bi_encoder", "cross_encoder", "unified")
    
    Returns:
        Normalized data in target format
    """
    normalized_data = []
    
    for item in data:
        if target_format == "bi_encoder":
            # Convert to bi-encoder format: {query, positive, negative, type}
            if "positive" in item and "negative" in item:
                normalized_data.append(item)
            elif "passage" in item and "label" in item:
                # Convert from cross-encoder format
                if item["label"] == 1.0:
                    # This is a positive example, need to find negative
                    normalized_data.append({
                        "query": item["query"],
                        "positive": item["passage"],
                        "negative": "",  # Will be filled later
                        "type": "triplet"
                    })
        
        elif target_format == "cross_encoder":
            # Convert to cross-encoder format: {query, passage, label, type}
            if "passage" in item and "label" in item:
                normalized_data.append(item)
            elif "positive" in item and "negative" in item:
                # Convert from bi-encoder format
                normalized_data.extend([
                    {
                        "query": item["query"],
                        "passage": item["positive"],
                        "label": 1.0,
                        "type": "positive_pair"
                    },
                    {
                        "query": item["query"],
                        "passage": item["negative"],
                        "label": 0.0,
                        "type": "negative_pair"
                    }
                ])
    
    return normalized_data


def generate_training_examples(
    train_data: List[Dict],
    processed_corpus: Dict[str, str],
    aid_map: Dict[int, str],
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Generate training examples (triplets and pairs) from the training data.

    Args:
        train_data: List of training questions.
        processed_corpus: Dictionary mapping canonical AIDs to content.
        aid_map: Dictionary mapping raw integer AIDs to canonical string AIDs.

    Returns:
        A tuple containing bi-encoder examples, cross-encoder examples, and stats.
    """
    logger.info("Generating training examples...")
    bi_encoder_examples = []
    cross_encoder_examples = []
    stats = {
        "questions_processed": 0,
        "questions_skipped_missing_aids": 0,
        "questions_without_valid_positives": 0,
        "total_positives_found": 0,
    }

    all_canonical_aids = list(processed_corpus.keys())

    for item in train_data:
        question = item.get("question")
        raw_aids = item.get("relevant_laws", [])  # Use the correct field name

        if not question or not raw_aids:
            stats["questions_skipped_missing_aids"] += 1
            continue

        stats["questions_processed"] += 1
        # Clean newline characters from question
        cleaned_question = (
            LegalTextCleaner()(question).replace("\\n", " ").replace("\\r", " ")
        )

        positive_canonical_aids = []
        for raw_aid in raw_aids:
            canonical_aid = aid_map.get(raw_aid)
            if canonical_aid and canonical_aid in processed_corpus:
                positive_canonical_aids.append(canonical_aid)

        if not positive_canonical_aids:
            stats["questions_without_valid_positives"] += 1
            continue

        stats["total_positives_found"] += len(positive_canonical_aids)

        # For each positive, create training examples
        for pos_aid in positive_canonical_aids:
            # Hard Negative Mining: Find negatives from the same law document for better difficulty
            pos_law_id = pos_aid.split("_")[0]  # Extract law_id from canonical_aid
            
            # First try to find negatives from the same law document
            same_law_negatives = [
                aid for aid in all_canonical_aids 
                if aid.startswith(pos_law_id) and aid not in positive_canonical_aids
            ]
            
            if same_law_negatives:
                # Use same law negative for harder examples
                negative_aid = random.choice(same_law_negatives)
            else:
                # Fallback to random negative if no same law negatives
                negative_aid = random.choice(
                    [
                        aid
                        for aid in all_canonical_aids
                        if aid not in positive_canonical_aids
                    ]
                )

            # Clean newline characters from content
            pos_content = (
                processed_corpus[pos_aid].replace("\\n", " ").replace("\\r", " ")
            )
            neg_content = (
                processed_corpus[negative_aid].replace("\\n", " ").replace("\\r", " ")
            )

            # Create bi-encoder training example (triplet)
            bi_encoder_example = {
                "query": cleaned_question,
                "positive": pos_content,
                "negative": neg_content,
                "type": "triplet"
            }
            bi_encoder_examples.append(bi_encoder_example)

            # Create cross-encoder training example (pair with label)
            cross_encoder_example = {
                "query": cleaned_question,
                "passage": pos_content,
                "label": 1.0,  # Positive example
                "type": "positive_classification"
            }
            cross_encoder_examples.append(cross_encoder_example)

            # Add negative example for cross-encoder
            cross_encoder_negative = {
                "query": cleaned_question,
                "passage": neg_content,
                "label": 0.0,  # Negative example
                "type": "negative_classification"
            }
            cross_encoder_examples.append(cross_encoder_negative)

    logger.info(f"Generated {len(bi_encoder_examples)} bi-encoder examples")
    logger.info(f"Generated {len(cross_encoder_examples)} cross-encoder examples")
    logger.info(f"Stats: {stats}")

    return bi_encoder_examples, cross_encoder_examples, stats


def process_legal_corpus(
    raw_corpus: List[Dict],
) -> Tuple[Dict[str, str], Dict[int, str]]:
    """Processes the raw legal corpus into a content dictionary and an AID map.

    Args:
        raw_corpus: A list of law documents from the raw JSON file.

    Returns:
        A tuple containing:
        - processed_corpus: A dictionary mapping canonical AIDs to article content.
        - aid_map: A dictionary mapping raw integer AIDs to canonical string AIDs.
    """
    logger.info("Processing legal corpus...")
    processed_corpus = {}
    aid_map = {}
    skipped_count = 0

    for law_doc in raw_corpus:
        law_id = law_doc.get("law_id")
        articles = law_doc.get("content", [])

        if not law_id or not isinstance(articles, list):
            continue

        for article in articles:
            raw_aid = article.get("aid")
            content = article.get("content_Article", "").strip()

            if not raw_aid or not content:
                skipped_count += 1
                continue

            try:
                canonical_aid = canonicalize_aid(str(law_id), str(raw_aid))
                processed_corpus[canonical_aid] = content
                # Ensure raw_aid is an integer for the map key
                aid_map[int(raw_aid)] = canonical_aid
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not process article with law_id='{law_id}', raw_aid='{raw_aid}': {e}"
                )
                skipped_count += 1

    logger.info(
        f"Processed {len(
    processed_corpus)} articles, created {len(aid_map)} AID mappings."
    )
    logger.info(f"Skipped {skipped_count} invalid or incomplete articles.")
    return processed_corpus, aid_map


def run_data_preparation():
    """Run the complete data preparation pipeline."""
    logger.info("🚀 Starting LawBot data preparation...")

    try:
        # Load raw data
        logger.info("Loading raw data...")
        train_data = load_json(config.paths.train_data_path)
        raw_corpus = load_json(config.paths.legal_corpus_path)

        logger.info(f"Loaded {len(train_data)} training questions")
        logger.info(f"Loaded {len(raw_corpus)} law documents")

        # Process corpus and create AID map
        processed_corpus, aid_map = process_legal_corpus(raw_corpus)

        # Generate training examples
        bi_encoder_examples, cross_encoder_examples, stats = generate_training_examples(
            train_data, processed_corpus, aid_map
        )

        # Create additional training data for Tier 2 (Light Ranking)
        light_ranking_examples = []
        negative_pool = []

        # Create training data for light ranking (query-passage pairs with labels)
        for item in bi_encoder_examples:
            # Positive example
            light_ranking_examples.append(
                {
                    "query": item["query"],
                    "passage": item["positive"],
                    "label": 1.0,
                    "type": "positive_pair",
                }
            )
            
            # Negative example
            light_ranking_examples.append(
                {
                    "query": item["query"],
                    "passage": item["negative"],
                    "label": 0.0,
                    "type": "negative_pair",
                }
            )

            # Add to negative pool for hard negative mining
            negative_pool.append(
                {"text": item["negative"], "type": "negative_candidate"}
            )

        # Add more negative candidates from processed corpus
        all_articles = list(processed_corpus.values())
        for i, article in enumerate(all_articles[:100]):  # Limit to 100 for performance
            if article not in [item["positive"] for item in bi_encoder_examples]:
                negative_pool.append({"text": article, "type": "corpus_negative"})

        # Create output directory
        output_dir = generate_versioned_path(config.paths.feature_dir, "processed_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        logger.info("Saving processed data...")
        save_jsonl(bi_encoder_examples, output_dir / "bi_encoder_train.jsonl")
        save_jsonl(cross_encoder_examples, output_dir / "cross_encoder_train.jsonl")
        save_json(processed_corpus, output_dir / "processed_corpus.json")

        # Save additional data for Tier 2
        save_jsonl(light_ranking_examples, output_dir / "training_data.jsonl")
        save_jsonl(negative_pool, output_dir / "negative_pool.jsonl")

        # Save metadata
        metadata = {
            "version": "8.0.0",
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(bi_encoder_examples)
            + len(cross_encoder_examples)
            + len(light_ranking_examples),
            "bi_encoder_examples": len(bi_encoder_examples),
            "cross_encoder_examples": len(cross_encoder_examples),
            "light_ranking_examples": len(light_ranking_examples),
            "negative_pool_size": len(negative_pool),
            "processed_articles": len(processed_corpus),
            "stats": stats,
        }
        save_metadata(output_dir, metadata)

        logger.info(f"✅ Data preparation completed successfully!")
        logger.info(f"Output directory: {output_dir}")

        return output_dir

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise


def main():
    """Main entry point."""
    try:
        output_dir = run_data_preparation()
        print(f"Data preparation completed. Output: {output_dir}")

    except Exception as e:
        print(f"Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
