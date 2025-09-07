#!/usr/bin/env python3
"""
Retrieval Engine - LawBot v8.1 (Refactored)
===========================================

Retrieval component that uses bi-encoder models and a FAISS index to find
relevant legal documents. It now self-manages content lookups.
"""

import json
from typing import List, Dict, Any
import logging
from pathlib import Path

try:
    import torch
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    from core.utils.logging_manager import get_logger
    from core.utils.parent_law_manager import ensure_parent_law_mapping
except ImportError as e:
    logging.error(f"Import error in retrieval.py: {e}")
    raise

logger = get_logger(__name__)


def _safe_move_to_device(model, device):
    """Safely move model to device, handling meta tensors and memory issues."""
    try:
        # Check GPU memory before moving to device
        if device != 'cpu' and torch.cuda.is_available():
            # Get current GPU memory usage
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free_memory = total_memory - current_memory
            
            logger.info(f"GPU Memory - Used: {current_memory:.2f}GB, Free: {free_memory:.2f}GB, Total: {total_memory:.2f}GB")
            
            # If less than 1GB free, clear cache and try again
            if free_memory < 1.0:
                logger.warning("Low GPU memory detected, clearing cache...")
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated() / 1024**3
                free_memory = total_memory - current_memory
                logger.info(f"After cache clear - Free: {free_memory:.2f}GB")
                
                # If still low memory, fallback to CPU
                if free_memory < 0.5:
                    logger.warning("Insufficient GPU memory, falling back to CPU")
                    return model.to('cpu')
        
        model.to(device)
        logger.info(f"✅ Successfully moved model to {device}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"CUDA out of memory, falling back to CPU: {e}")
            torch.cuda.empty_cache()  # Clear cache before fallback
            return model.to('cpu')
        else:
            raise
    except NotImplementedError as e:
        if "meta tensor" in str(e).lower():
            logger.info(f"Detected meta tensor, using to_empty() for device: {device}")
            model.to_empty(device=device)
        else:
            raise
    return model


class RetrievalEngine:
    """Retrieval engine using a bi-encoder and FAISS index."""

    def __init__(
        self,
        bi_encoder_path: str,
        faiss_index_path: str,
        content_map_path: str,
        index_to_aid_path: str,
    ):
        """Initializes the retrieval engine with smart memory management."""
        # Determine device with memory check
        self.device = self._determine_optimal_device()
        self.is_ready = False
        logger.info(f"Using device: {self.device}")

        try:
            logger.info(f"Loading bi-encoder from: {bi_encoder_path}")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cleared GPU cache before model loading")
            
            # Load SentenceTransformer with memory-safe approach
            try:
                # Load model on CPU first to avoid memory issues
                logger.info("🔄 Loading SentenceTransformer on CPU first...")
                self.bi_encoder = SentenceTransformer(bi_encoder_path, device='cpu')
                logger.info("✅ Successfully loaded SentenceTransformer on CPU")
                
                # Now safely move to target device
                if self.device != 'cpu':
                    logger.info(f"🔄 Moving SentenceTransformer to {self.device}...")
                    self.bi_encoder = _safe_move_to_device(self.bi_encoder, self.device)
                    logger.info(f"✅ Successfully moved SentenceTransformer to {self.device}")
                else:
                    logger.info("ℹ️ SentenceTransformer kept on CPU")
                
                # Verify model is ready and log memory usage
                self._log_memory_usage("After model loading")
                
            except Exception as e:
                logger.error(f"❌ Failed to load SentenceTransformer: {e}")
                # Try fallback to CPU if GPU failed
                if self.device != 'cpu':
                    logger.warning("🔄 Attempting fallback to CPU...")
                    try:
                        self.bi_encoder = SentenceTransformer(bi_encoder_path, device='cpu')
                        self.device = 'cpu'
                        logger.info("✅ Fallback to CPU successful")
                    except Exception as fallback_e:
                        logger.error(f"❌ CPU fallback also failed: {fallback_e}")
                        raise
                else:
                    raise

            logger.info(f"Loading FAISS index from: {faiss_index_path}")
            self.faiss_index = faiss.read_index(faiss_index_path)

            logger.info(f"Loading content map from: {content_map_path}")
            with open(content_map_path, "r", encoding="utf-8") as f:
                self.corpus_content = json.load(f)

            logger.info(f"Loading index-to-AID map from: {index_to_aid_path}")
            with open(index_to_aid_path, "r", encoding="utf-8") as f:
                self.index_to_aid = json.load(f)

            # Load AID to parent law mapping
            # Use utility to ensure mapping exists and is valid
            logger.info("Ensuring parent law mapping is available...")
            if ensure_parent_law_mapping():
                aid_to_parent_path = Path("features/aid_to_parent_law.json")
                with open(aid_to_parent_path, "r", encoding="utf-8") as f:
                    self.aid_to_parent_law = json.load(f)
                logger.info(f"Loaded {len(self.aid_to_parent_law)} parent law mappings")
                # Log a few sample mappings for verification
                sample_mappings = list(self.aid_to_parent_law.items())[:3]
                logger.info(f"Sample mappings: {sample_mappings}")
            else:
                logger.warning(
                    "Failed to ensure parent law mapping, parent law names will not be available"
                )
                self.aid_to_parent_law = {}

            self.is_ready = True
            logger.info(
                f"Retrieval engine initialized successfully. Index size: {self.faiss_index.ntotal}"
            )
        except FileNotFoundError as e:
            logger.error(f"A required mapping file was not found: {e}")
            self.is_ready = False
            raise
        except Exception as e:
            logger.error(f"Failed to initialize retrieval engine: {e}", exc_info=True)
            self.is_ready = False
            raise

    def _determine_optimal_device(self) -> str:
        """Determine the optimal device based on available memory."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return "cpu"
        
        try:
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            free_memory = total_memory - current_memory
            
            logger.info(f"GPU Memory - Total: {total_memory:.2f}GB, Used: {current_memory:.2f}GB, Free: {free_memory:.2f}GB")
            
            # Need at least 2GB free for bi-encoder model
            if free_memory >= 2.0:
                logger.info("✅ Sufficient GPU memory available")
                return "cuda"
            else:
                logger.warning(f"⚠️ Insufficient GPU memory ({free_memory:.2f}GB < 2.0GB), using CPU")
                return "cpu"
                
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}, falling back to CPU")
            return "cpu"
    
    def _log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"Memory usage {context} - Current: {current_memory:.2f}GB, Peak: {max_memory:.2f}GB")
        else:
            logger.info(f"Memory usage {context} - CPU mode")

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieves relevant documents for a single query."""
        # Reuse the batch method for a single query
        results = self.retrieve_batch([query], top_k=top_k)
        return results[0] if results else []

    def retrieve_batch(
        self, queries: List[str], top_k: int = 100
    ) -> List[List[Dict[str, Any]]]:
        """Retrieves relevant documents for a batch of queries with memory optimization."""
        if not self.is_ready:
            raise RuntimeError("Retrieval engine is not ready.")
        if not queries:
            return []

        try:
            # Process in smaller batches to avoid memory issues
            batch_size = min(len(queries), 32)  # Limit batch size
            all_results = []
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size}")
                
                # Encode queries in a batch
                query_embeddings = self.bi_encoder.encode(
                    batch_queries, 
                    convert_to_tensor=True,
                    show_progress_bar=False  # Disable progress bar for cleaner logs
                )
                query_embeddings_np = query_embeddings.cpu().numpy()
                faiss.normalize_L2(query_embeddings_np)
                
                # Clear GPU cache after encoding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Search in a batch
                scores_batch, indices_batch = self.faiss_index.search(
                    query_embeddings_np, top_k
                )

                batch_results = []
                for j, query in enumerate(batch_queries):
                    scores = scores_batch[j]
                    indices = indices_batch[j]

                    results = []
                    for rank, (score, idx) in enumerate(zip(scores, indices)):
                        if idx < 0:  # Invalid index
                            continue

                        aid = self.index_to_aid.get(str(idx))
                        # Convert AID to string for corpus_content lookup
                        aid_str = str(aid) if aid is not None else None
                        content_info = self.corpus_content.get(aid_str)

                        if content_info:
                            # Get parent law name from mapping
                            parent_law_name = self.aid_to_parent_law.get(
                                str(aid), "Unknown"
                            )

                            # Debug logging for parent law lookup (only for first few)
                            if parent_law_name == "Unknown" and j < 3:
                                logger.debug(
                                    f"Parent law not found for AID {aid} (type: {type(aid)})"
                                )
                            elif j < 3:
                                logger.debug(
                                    f"Found parent law '{parent_law_name}' for AID {aid}"
                                )

                            results.append(
                                {
                                    "aid": aid,
                                    "parent_law_name": parent_law_name,
                                    "content": (
                                        content_info
                                        if isinstance(content_info, str)
                                        else str(content_info)
                                    ),
                                    "retrieval_score": float(score),
                                    "retrieval_rank": rank + 1,
                                }
                            )
                    batch_results.append(results)
                
                all_results.extend(batch_results)

            logger.debug(f"Retrieved documents for {len(queries)} queries.")
            return all_results
        except Exception as e:
            logger.error(f"Error in retrieve_batch: {e}", exc_info=True)
            return [[] for _ in queries]

    def get_index_info(self) -> Dict[str, Any]:
        """Gets information about the FAISS index."""
        if not self.is_ready:
            return {"status": "Engine not ready"}
        return {
            "index_size": self.faiss_index.ntotal,
            "index_dimension": self.faiss_index.d,
            "aid_mapping_size": len(self.index_to_aid),
            "content_map_size": len(self.corpus_content),
            "device": self.device,
        }

    def cleanup(self):
        """Cleans up resources used by the retrieval engine with thorough memory management."""
        try:
            # Log memory before cleanup
            self._log_memory_usage("Before cleanup")
            
            # Clean up model
            if hasattr(self, 'bi_encoder'):
                del self.bi_encoder
                logger.info("✅ Bi-encoder model cleaned up")
            
            # Clean up other resources
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
            if hasattr(self, 'corpus_content'):
                del self.corpus_content
            if hasattr(self, 'index_to_aid'):
                del self.index_to_aid
            if hasattr(self, 'aid_to_parent_law'):
                del self.aid_to_parent_law
            
            # Clear GPU cache multiple times for thorough cleanup
            if torch.cuda.is_available():
                for i in range(3):  # Multiple cache clears
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for operations to complete
                logger.info("✅ GPU cache cleared multiple times")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log memory after cleanup
            self._log_memory_usage("After cleanup")
            
            logger.info("✅ Retrieval engine cleanup completed successfully")
            
        except AttributeError:
            logger.info("ℹ️ Some attributes not found during cleanup (normal if init failed)")
        except Exception as e:
            logger.warning(f"⚠️ Error during retrieval engine cleanup: {e}")
        finally:
            self.is_ready = False


# Alias for backward compatibility
Retriever = RetrievalEngine
