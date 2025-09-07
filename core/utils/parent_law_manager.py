#!/usr/bin/env python3
"""
Parent Law Mapping Manager - LawBot v8.3
========================================

Utility module to manage parent law mapping creation and validation.
Ensures parent law mapping is always available for the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def create_parent_law_mapping() -> Optional[Dict[str, str]]:
    """
    Create parent law mapping from processed corpus and aid map.

    Returns:
        Dictionary mapping numeric AID to parent law name, or None if failed
    """
    try:
        # Find the latest processed data directory
        features_dir = Path("features")
        processed_dirs = [
            d
            for d in features_dir.iterdir()
            if d.is_dir() and d.name.startswith("processed_data_")
        ]

        if not processed_dirs:
            logger.error("No processed data directories found")
            return None

        # Use the most recent one
        latest_processed_dir = max(processed_dirs, key=lambda x: x.name)
        processed_corpus_path = latest_processed_dir / "processed_corpus.json"

        if not processed_corpus_path.exists():
            logger.error(f"Processed corpus not found at {processed_corpus_path}")
            return None

        # Load the aid_map to get numeric AIDs
        aid_map_path = Path("features/aid_map.json")
        if not aid_map_path.exists():
            logger.error(f"AID map not found at {aid_map_path}")
            return None

        logger.info(f"Creating parent law mapping from {processed_corpus_path}")

        # Load data
        with open(processed_corpus_path, "r", encoding="utf-8") as f:
            processed_corpus = json.load(f)

        with open(aid_map_path, "r", encoding="utf-8") as f:
            aid_map = json.load(f)

        # Create reverse mapping from content to numeric AID
        content_to_numeric_aid = {}
        for numeric_aid, content in aid_map.items():
            content_to_numeric_aid[content] = numeric_aid

        # Create mapping from numeric AID to parent law name
        aid_to_parent_law = {}

        for string_aid, content in processed_corpus.items():
            # Extract parent law name from string AID (e.g., "14/2022/TTNHNN_1" -> "14/2022/TTNHNN")
            if "_" in string_aid:
                parent_law_name = string_aid.split("_")[0]
            else:
                parent_law_name = string_aid

            # Find the numeric AID for this content
            if content in content_to_numeric_aid:
                numeric_aid = content_to_numeric_aid[content]
                aid_to_parent_law[numeric_aid] = parent_law_name

        logger.info(f"Created mapping for {len(aid_to_parent_law)} documents")
        return aid_to_parent_law

    except Exception as e:
        logger.error(f"Failed to create parent law mapping: {e}")
        return None


def ensure_parent_law_mapping() -> bool:
    """
    Ensure parent law mapping exists. Create if missing.

    Returns:
        True if mapping is available, False otherwise
    """
    mapping_path = Path("features/aid_to_parent_law.json")

    # Check if mapping already exists
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if mapping and len(mapping) > 0:
                logger.info(
                    f"Parent law mapping already exists with {len(mapping)} entries"
                )
                return True
        except Exception as e:
            logger.warning(f"Existing parent law mapping is corrupted: {e}")

    # Create new mapping
    logger.info("Parent law mapping not found or corrupted, creating new one...")
    mapping = create_parent_law_mapping()

    if mapping:
        try:
            # Save the mapping
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Parent law mapping created and saved to {mapping_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save parent law mapping: {e}")
            return False
    else:
        logger.error("Failed to create parent law mapping")
        return False


def get_parent_law_mapping() -> Optional[Dict[str, str]]:
    """
    Get parent law mapping, creating it if necessary.

    Returns:
        Dictionary mapping numeric AID to parent law name, or None if failed
    """
    if ensure_parent_law_mapping():
        try:
            mapping_path = Path("features/aid_to_parent_law.json")
            with open(mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load parent law mapping: {e}")
            return None
    return None
