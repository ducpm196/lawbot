#!/usr/bin/env python3
"""
LegalQA Pipeline - LawBot v8.1 (Refactored)
===========================================

Main pipeline for legal question answering that orchestrates retrieval and
reranking components using a simplified, unified architecture.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys
from datetime import datetime
import time

try:
    from core.retrieval import RetrievalEngine
    from core.reranking import RerankingEngine
    from core.utils.logging_manager import get_logger
    from core.utils.versioning import get_latest_version_path
    from core.utils.parent_law_manager import ensure_parent_law_mapping
    # Auto-evaluator removed for stability
    from config.loader import config
except ImportError as e:
    logging.error(f"Import error in pipeline.py: {e}")
    raise

logger = get_logger(__name__)


class LegalQAPipeline:
    """LegalQA Pipeline class for orchestrating retrieval and reranking."""

    def __init__(
        self,
        bi_encoder_path: Optional[Path] = None,
        reranker_paths: Optional[Dict[str, Path]] = None,
        faiss_index_path: Optional[Path] = None,
        content_map_path: Optional[Path] = None,
        index_to_aid_path: Optional[Path] = None,
        enable_auto_evaluation: bool = False,  # Disabled for stability
    ):
        """Initialize the LegalQA Pipeline."""
        self.is_ready = False
        self.loaded_model_paths = {}
        self.enable_auto_evaluation = enable_auto_evaluation
        
        # Initialize configuration
        self.config = self._initialize_config()
        
        # Auto-evaluation disabled for stability
        self.auto_evaluator = None

        try:
            # Ensure parent law mapping is available before proceeding
            logger.info("Validating parent law mapping availability...")
            if not ensure_parent_law_mapping():
                logger.warning(
                    "Parent law mapping validation failed, but continuing with pipeline initialization"
                )
            else:
                logger.info("✅ Parent law mapping validation successful")

            # --- Load Retriever ---
            paths = config.paths
            bi_encoder_to_load = bi_encoder_path or get_latest_version_path(
                paths.model_dir, "bi-encoder"
            )
            faiss_to_load = faiss_index_path or paths.faiss_index_path
            content_map_to_load = content_map_path or paths.content_map_path
            index_to_aid_to_load = index_to_aid_path or paths.index_to_aid_path

            if not all(
                [
                    bi_encoder_to_load,
                    faiss_to_load.exists(),
                    content_map_to_load.exists(),
                    index_to_aid_to_load.exists(),
                ]
            ):
                raise FileNotFoundError(
                    f"One or more required files for the retriever could not be found. "
                    f"Checked paths: Bi-Encoder: {bi_encoder_to_load}, "
                    f"FAISS: {faiss_to_load}, Content Map: {content_map_to_load}, "
                    f"Index Map: {index_to_aid_to_load}"
                )

            logger.info(f"Loading Retriever with Bi-Encoder: {bi_encoder_to_load}")
            try:
                self.retriever = RetrievalEngine(
                    bi_encoder_path=str(bi_encoder_to_load),
                    faiss_index_path=str(faiss_to_load),
                    content_map_path=str(content_map_to_load),
                    index_to_aid_path=str(index_to_aid_to_load),
                )
            except Exception as e:
                logger.error(f"Failed to initialize RetrievalEngine: {e}")
                raise
            self.loaded_model_paths["bi_encoder"] = bi_encoder_to_load
            self.loaded_model_paths["faiss_index"] = faiss_to_load

            # --- Load Unified Reranker ---
            reranker_configs_to_load = self._resolve_reranker_paths(reranker_paths)
            if not reranker_configs_to_load:
                logger.warning(
                    "No reranker models configured, continuing without reranking."
                )
                self.reranker = None
            else:
                self.reranker = RerankingEngine(
                    reranker_configs=reranker_configs_to_load
                )
                self.loaded_model_paths["rerankers"] = {
                    name: cfg["path"] for name, cfg in reranker_configs_to_load.items()
                }

            self.is_ready = True
            logger.info("LegalQAPipeline initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize LegalQAPipeline: {e}", exc_info=True)
            self.is_ready = False
            raise

    def get_loaded_model_versions(self) -> Dict[str, str]:
        """Returns a dictionary of loaded model names and their version IDs."""
        versions = {}
        if "bi_encoder" in self.loaded_model_paths:
            versions["Bi-Encoder"] = Path(self.loaded_model_paths["bi_encoder"]).name
        if "rerankers" in self.loaded_model_paths:
            for name, path in self.loaded_model_paths["rerankers"].items():
                versions[name.replace("_", " ").title()] = Path(path).name
        return versions

    def _resolve_reranker_paths(
        self, manual_paths: Optional[Dict[str, Path]]
    ) -> Dict[str, Any]:
        """Resolves paths for reranker models."""
        if manual_paths:
            logger.info(
                f"Loading rerankers from manually provided paths: {manual_paths}"
            )
            return {
                name: {"path": path, "enabled": True}
                for name, path in manual_paths.items()
            }

        logger.info("Auto-discovering latest reranker model versions...")
        resolved_configs = {}

        # Map config names to actual model directory names
        reranker_model_mapping = {
            "light_reranker": "light-ranking",  # SentenceTransformer model
            "cross_encoder": "combined-reranker-adapt",  # Ensemble model
        }

        # Map config names to actual model subdirectories (for models with subdirectory structure)
        # Note: cross_encoder is an ensemble model loaded from root directory
        reranker_subdir_mapping = {
            # "cross_encoder": "adapt_enhanced",  # Cross-encoder model is loaded from root directory
        }

        reranker_models = {
            "light_reranker": config.reranker_pipeline.light_reranker,
            "cross_encoder": config.reranker_pipeline.cross_encoder,
        }

        for config_name, model_cfg in reranker_models.items():
            if (
                model_cfg
                and isinstance(model_cfg, dict)
                and model_cfg.get("enabled", False)
            ):
                # Get the actual model directory name
                model_dir_name = reranker_model_mapping.get(config_name, config_name)
                latest_path = get_latest_version_path(
                    config.paths.model_dir, model_dir_name
                )
                if latest_path:
                    # Check if this model has a subdirectory structure
                    if config_name in reranker_subdir_mapping:
                        subdir = reranker_subdir_mapping[config_name]
                        model_path = latest_path / subdir
                        if model_path.exists():
                            latest_path = model_path
                            logger.info(
                                f"Using subdirectory {subdir} for {config_name}: {model_path}"
                            )
                        else:
                            logger.warning(
                                f"Subdirectory {subdir} not found for {config_name} at {model_path}"
                            )

                    # Create a dictionary from the config
                    resolved_configs[config_name] = model_cfg.copy()
                    resolved_configs[config_name]["path"] = str(latest_path)

                    # Add model type information for proper handling
                    if config_name == "light_reranker":
                        resolved_configs[config_name][
                            "model_type"
                        ] = "sentence_transformer"
                    else:
                        resolved_configs[config_name]["model_type"] = "classification"

                    logger.info(
                        f"Found reranker {config_name} ({model_dir_name}) at: {latest_path}"
                    )
                else:
                    logger.warning(
                        f"Reranker {config_name} ({model_dir_name}) is enabled but no model found"
                    )
            else:
                logger.info(f"Reranker {config_name} is disabled or not configured")
        return resolved_configs
    
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize pipeline configuration with defaults."""
        return {
            "top_k_retrieval": 20,
            "top_k_light": 10,
            "use_light_ranking": True,
            "use_cross_encoder": True,
            "light_weight": 0.3,
            "cross_encoder_weight": 0.7
        }
    
    def predict(
        self,
        query: str,
        top_k: int = 5,
        include_scores: bool = True,
        include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """Predict relevant documents for a query."""
        start_time = time.time()
        
        try:
            # Tier 1: Bi-Encoder Retrieval
            retrieval_results = self._retrieve_documents(query, top_k=self.config["top_k_retrieval"])
            
            if not retrieval_results:
                logger.warning("No documents retrieved from Tier 1")
                return []
            
            # Tier 2: Light Reranker (if enabled)
            if self.config["use_light_ranking"] and self.reranker:
                light_results = self._light_rerank(query, retrieval_results, top_k=self.config["top_k_light"])
            else:
                light_results = retrieval_results
            
            # Tier 3: Cross-Encoder (if enabled)
            if self.config["use_cross_encoder"] and self.reranker:
                final_results = self._cross_encode_rerank(query, light_results, top_k)
            else:
                final_results = light_results[:top_k]
            
            # Add metadata and scores
            results = self._add_metadata(final_results, include_scores, include_content)
            
            # Auto-evaluation (if enabled)
            if self.auto_evaluator:
                try:
                    # Get pipeline configuration for evaluation
                    pipeline_config = {
                        "top_k_retrieval": self.config["top_k_retrieval"],
                        "top_k_light": self.config["top_k_light"],
                        "top_k_final": top_k,
                        "use_light_ranking": self.config["use_light_ranking"],
                        "use_cross_encoder": self.config["use_cross_encoder"],
                        "light_weight": self.config["light_weight"],
                        "cross_encoder_weight": self.config["cross_encoder_weight"]
                    }
                    
                    # Get model versions
                    model_versions = self._get_model_versions()
                    
                    # Performance tracking (simplified, no auto-evaluation)
                    logger.info(f"Query processed with {len(results)} results")
                    
                except Exception as e:
                    logger.warning(f"Auto-evaluation failed: {e}")
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"Query completed in {query_time:.1f}ms, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the current status of the pipeline."""
        status = {
            "is_ready": self.is_ready,
            "loaded_models": self.get_loaded_model_versions(),
            "retriever_ready": hasattr(self, "retriever") and self.retriever.is_ready,
            "reranker_ready": False,
            "auto_evaluation_enabled": self.enable_auto_evaluation,
        }
        if hasattr(self, "reranker") and self.reranker is not None:
            status["reranker_ready"] = self.reranker.is_ready
            status["reranker_info"] = self.reranker.get_engine_info()
        
        # Auto-evaluation status (disabled for stability)
        status["auto_evaluation"] = {
            "status": "disabled",
            "reason": "Removed for system stability"
        }
        
        return status

    def cleanup(self):
        """Clean up resources used by the pipeline."""
        try:
            if hasattr(self, "retriever"):
                self.retriever.cleanup()
            if hasattr(self, "reranker"):
                self.reranker.cleanup()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")
    
    def export_evaluation_data(self, output_path: Optional[Path] = None) -> bool:
        """Export evaluation data for analysis."""
        logger.warning("Auto-evaluation export disabled for stability")
        return False
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary for monitoring."""
        return {
            "status": "disabled",
            "reason": "Auto-evaluation removed for system stability"
        }

    def _retrieve_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve documents using Bi-Encoder."""
        try:
            documents = self.retriever.retrieve(query, top_k=top_k)
            
            # Initialize scores
            for doc in documents:
                doc["light_reranker_score"] = 0.0
                doc["cross_encoder_score"] = 0.0
                doc["phobert_base_score"] = 0.0
                doc["phobert_large_score"] = 0.0
                doc["tier3_ensemble_score"] = 0.0
            
            return documents
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _light_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Apply light reranking."""
        try:
            if not self.reranker or not self.reranker.is_ready:
                logger.warning("Light reranker not available, skipping")
                return documents
            
            reranked = self.reranker.rank_light(query, documents[:top_k])
            return reranked
        except Exception as e:
            logger.error(f"Light reranking failed: {e}")
            return documents
    
    def _cross_encode_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking."""
        try:
            if not self.reranker or not self.reranker.is_ready:
                logger.warning("Cross-encoder not available, skipping")
                return documents[:top_k]
            
            # Apply cross-encoder reranking
            reranked = self.reranker.rank_cross(query, documents)
            
            # Calculate ensemble scores for Tier 3
            for doc in reranked:
                phobert_base_score = doc.get("phobert_base_score", 0.0)
                phobert_large_score = doc.get("phobert_large_score", 0.0)
                
                # Ensemble score (70% PhoBERT-base + 30% PhoBERT-large)
                tier3_ensemble_score = (0.7 * phobert_base_score + 0.3 * phobert_large_score)
                doc["tier3_ensemble_score"] = tier3_ensemble_score
                
                # Calculate final score
                retrieval_score = doc.get("retrieval_score", 0.0)
                light_score = doc.get("light_reranker_score", 0.0)
                
                final_score = retrieval_score
                if self.config["use_light_ranking"]:
                    final_score = (self.config["light_weight"] * light_score + 
                                 (1 - self.config["light_weight"]) * final_score)
                if self.config["use_cross_encoder"]:
                    final_score = (self.config["cross_encoder_weight"] * tier3_ensemble_score + 
                                 (1 - self.config["cross_encoder_weight"]) * final_score)
                
                doc["final_score"] = final_score
                doc["score_breakdown"] = {
                    "retrieval": retrieval_score,
                    "light_reranker": light_score,
                    "tier3_ensemble": tier3_ensemble_score,
                    "phobert_base": phobert_base_score,
                    "phobert_large": phobert_large_score,
                }
            
            # Sort by final score and return top_k
            return sorted(reranked, key=lambda x: x["final_score"], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents[:top_k]
    
    def _add_metadata(self, documents: List[Dict[str, Any]], include_scores: bool, include_content: bool) -> List[Dict[str, Any]]:
        """Add metadata to documents."""
        for doc in documents:
            if not include_scores:
                # Remove score fields if not requested
                doc.pop("retrieval_score", None)
                doc.pop("light_reranker_score", None)
                doc.pop("cross_encoder_score", None)
                doc.pop("final_score", None)
                doc.pop("score_breakdown", None)
            
            if not include_content:
                # Remove content if not requested
                doc.pop("content", None)
        
        return documents
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get loaded model versions for evaluation."""
        versions = {}
        
        if hasattr(self, "retriever") and self.retriever:
            versions["Bi-Encoder"] = getattr(self.retriever, 'model_version', 'Unknown')
        
        if hasattr(self, "reranker") and self.reranker:
            versions["Light Reranker"] = getattr(self.reranker, 'light_model_version', 'Unknown')
            versions["Cross Encoder"] = getattr(self.reranker, 'cross_model_version', 'Unknown')
        
        return versions


# Alias for backward compatibility
Pipeline = LegalQAPipeline
