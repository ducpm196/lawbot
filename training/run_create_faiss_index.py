#!/usr/bin/env python3
"""
FAISS Index Creation - LawBot v8.2 Training Pipeline
====================================================

Creates FAISS index using the trained Bi-Encoder model as part of the training pipeline.
This ensures consistency and proper model usage throughout the pipeline.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from core.utils.logging_manager import setup_logging
    from core.utils.versioning import generate_versioned_path, save_metadata
    from core.utils.io import load_jsonl, save_json
    from core.utils.system_check import get_device_info
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# SỬA: Use centralized workflow logging (no duplicate setup)
from core.utils.logging_manager import get_workflow_timestamp
workflow_timestamp = get_workflow_timestamp()
if workflow_timestamp:
    # Use existing workflow logging
    logger = logging.getLogger(__name__)
else:
    # Setup new workflow logging
    setup_logging("workflow")
    logger = logging.getLogger(__name__)


def create_faiss_index():
    """Create FAISS index using the trained Bi-Encoder model."""

    # Paths
    project_root = Path(__file__).parent.parent
    # Find the latest bi-encoder model
    bi_encoder_models = list(project_root.glob("models/bi-encoder_*"))
    if not bi_encoder_models:
        logger.error("No Bi-Encoder models found")
        return False

    bi_encoder_path = max(bi_encoder_models, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using Bi-Encoder model: {bi_encoder_path}")
    legal_corpus_path = project_root / "data" / "raw" / "legal_corpus.json"
    features_dir = project_root / "features"

    # Check if Bi-Encoder model exists
    if not bi_encoder_path.exists():
        logger.error(f"Bi-Encoder model not found at: {bi_encoder_path}")
        return False

    # Check if legal corpus exists
    if not legal_corpus_path.exists():
        logger.error(f"Legal corpus not found at: {legal_corpus_path}")
        return False

    try:
        logger.info("🔍 Loading trained Bi-Encoder model...")
        model = SentenceTransformer(str(bi_encoder_path))

        logger.info("📚 Loading legal corpus...")
        with open(legal_corpus_path, "r", encoding="utf-8") as f:
            legal_corpus = json.load(f)

        logger.info(f"📊 Processing {len(legal_corpus)} documents...")

        # Extract content and create AID mapping
        documents = []
        aid_map = {}
        index_to_aid = {}

        for i, article in enumerate(legal_corpus):
            if isinstance(article, dict) and "content" in article:
                # content is a list of articles
                if isinstance(article["content"], list):
                    for j, sub_article in enumerate(article["content"]):
                        if (
                            isinstance(sub_article, dict)
                            and "aid" in sub_article
                            and "content_Article" in sub_article
                        ):
                            content = sub_article["content_Article"]
                            aid = sub_article["aid"]
                            if (
                                content
                                and isinstance(content, str)
                                and len(content.strip()) > 0
                            ):
                                documents.append(content)
                                aid_map[aid] = content
                                index_to_aid[str(len(documents) - 1)] = aid

        logger.info(f"✅ Processed {len(documents)} valid documents")

        if len(documents) > 0:
            logger.info("🚀 Creating document embeddings...")

            # Filter documents by length to avoid CUDA errors
            max_length = 512
            filtered_documents = []
            filtered_aid_map = {}
            filtered_index_to_aid = {}

            for i, doc in enumerate(documents):
                if len(doc) <= max_length:
                    filtered_documents.append(doc)
                    aid = list(aid_map.keys())[i]
                    filtered_aid_map[aid] = doc
                    filtered_index_to_aid[str(len(filtered_documents) - 1)] = aid

            logger.info(
                f"📊 Filtered to {len(filtered_documents)} documents (max length: {max_length})"
            )

            # Process in smaller batches
            batch_size = 16
            all_embeddings = []
            total_batches = (len(filtered_documents) + batch_size - 1) // batch_size

            logger.info(
                f"🚀 Processing {len(filtered_documents)} documents in {total_batches} batches..."
            )

            for i in range(0, len(filtered_documents), batch_size):
                batch = filtered_documents[i : i + batch_size]
                current_batch = i // batch_size + 1

                # Create progress bar
                progress = current_batch / total_batches
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                percentage = progress * 100

                # Print progress bar on same line
                print(
                    f"\r🔄 Processing: [{bar}] {percentage:5.1f}% ({current_batch}/{total_batches})",
                    end="",
                    flush=True,
                )

                try:
                    batch_embeddings = model.encode(
                        batch, show_progress_bar=False, batch_size=8
                    )
                    all_embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.warning(f"\n⚠️ Error processing batch {current_batch}: {e}")
                    continue

            # Print newline after progress bar
            print()

            if all_embeddings:
                embeddings = np.vstack(all_embeddings)
                logger.info(f"✅ Created embeddings: {embeddings.shape}")

                # Update mappings for filtered documents
                aid_map = filtered_aid_map
                index_to_aid = filtered_index_to_aid
                documents = filtered_documents
            else:
                logger.error("❌ No embeddings created!")
                return False

            # Normalize embeddings
            faiss.normalize_L2(embeddings)

            # Create FAISS index
            logger.info("🔍 Creating FAISS index...")
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            index.add(embeddings)

            # Save index and mappings
            logger.info("💾 Saving FAISS index and mappings...")

            # Save FAISS index
            faiss_index_path = features_dir / "faiss_index.bin"
            faiss.write_index(index, str(faiss_index_path))

            # Save AID map
            aid_map_path = features_dir / "aid_map.json"
            with open(aid_map_path, "w", encoding="utf-8") as f:
                json.dump(aid_map, f, ensure_ascii=False, indent=2)

            # Save index to AID mapping
            index_to_aid_path = features_dir / "index_to_aid.json"
            with open(index_to_aid_path, "w", encoding="utf-8") as f:
                json.dump(index_to_aid, f, ensure_ascii=False, indent=2)

            # Save metadata
            metadata = {
                "index_type": "IndexFlatIP",
                "dimension": dimension,
                "total_documents": len(documents),
                "bi_encoder_model": str(bi_encoder_path),
                "created_at": "2025-08-16_22:15:00",
                "description": "FAISS index created with trained Bi-Encoder model as part of training pipeline",
            }

            metadata_path = features_dir / "faiss_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info("✅ FAISS index creation completed successfully!")
            logger.info(f"📁 Index saved to: {faiss_index_path}")
            logger.info(f"📁 AID map saved to: {aid_map_path}")
            logger.info(f"📁 Index mapping saved to: {index_to_aid_path}")
            logger.info(f"📁 Metadata saved to: {metadata_path}")
            logger.info(f"📊 Final index size: {index.ntotal} documents")
            logger.info(f"📊 Embedding dimension: {dimension}")

            return True

        else:
            logger.error("❌ No valid documents found!")
            return False

    except Exception as e:
        logger.error(f"❌ Error creating FAISS index: {e}")
        return False


if __name__ == "__main__":
    success = create_faiss_index()
    if success:
        print("🎉 FAISS index creation completed successfully!")
    else:
        print("❌ FAISS index creation failed!")
        exit(1)
