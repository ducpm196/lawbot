"""
Centralized Model Configuration for LawBot
========================================

This module provides centralized configuration for all model names, types, and mappings
used throughout the system. This makes it easier to manage and update model configurations
without having to modify multiple files.
"""

from typing import Dict, List, Any
from pathlib import Path

# --- Model Type Definitions ---
MODEL_TYPES = {
    "bi_encoder": {
        "name": "Bi-Encoder",
        "display_name": "Bi-Encoder Retrieval",
        "directory_prefix": "bi-encoder",
        "model_name": "bkai-foundation-models/vietnamese-bi-encoder",
        "purpose": "Fast retrieval with Vietnamese domain expertise",
        "tier": "tier_1",
        "performance": "~1000+ docs/second",
        "techniques": ["Contrastive Learning", "ADAPT", "HPO", "HNM"],
    },
    "light_reranker": {
        "name": "Light Reranker",
        "display_name": "Light Reranking",
        "directory_prefix": "light-ranking",
        "model_name": "vinai/phobert-base-v2",
        "purpose": "Fast filtering with independent domain expertise",
        "tier": "tier_2",
        "performance": "~500+ docs/second",
        "techniques": ["HPO", "HNM", "Independent ADAPT"],
    },
    "cross_encoder": {
        "name": "Cross-Encoder",
        "display_name": "Cross-Encoder Ensemble",
        "directory_prefix": "combined-reranker-adapt",
        "model_name": "vinai/phobert-large",
        "purpose": "Final ranking with domain expertise + general quality balance",
        "tier": "tier_3",
        "performance": "~100+ docs/second",
        "techniques": ["HPO", "HNM", "Ensemble", "ADAPT-enhanced inheritance"],
    },
}

# --- Model Directory Mappings ---
MODEL_DIRECTORY_MAPPING = {
    "bi_encoder": "bi-encoder",
    "light_reranker": "light-ranking",
    "cross_encoder": "combined-reranker-adapt",
}

# --- Model Status Keys ---
MODEL_STATUS_KEYS = {
    "bi_encoder": "bi_encoder",
    "light_reranker": "light_reranker",
    "cross_encoder": "cross_encoder",
}

# --- Training Stage Names ---
TRAINING_STAGES = {
    "bi_encoder": "bi_encoder",
    "light_reranker": "light_ranking",
    "cross_encoder": "cross_encoder",
}

# --- Data Type Mappings ---
DATA_TYPE_MAPPINGS = {
    "bi_encoder": "bi_encoder",
    "light_reranker": "training_data",
    "cross_encoder": "cross_encoder",
}

# --- Score Field Names ---
SCORE_FIELD_NAMES = {
    "bi_encoder": "retrieval_score",
    "light_reranker": "light_reranker_score",
    "cross_encoder": "cross_encoder_score",
}

# --- Display Names for UI ---
DISPLAY_NAMES = {
    "bi_encoder": "Điểm retrieval",
    "light_reranker": "Điểm light rerank",
    "cross_encoder": "Điểm cross encoder",
}

# --- Model Loading Types ---
MODEL_LOADING_TYPES = {
    "bi_encoder": "sentence_transformer",
    "light_reranker": "sentence_transformer",
    "cross_encoder": "classification",
}


# --- Helper Functions ---
def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get configuration for a specific model type."""
    return MODEL_TYPES.get(model_key, {})


def get_all_model_keys() -> List[str]:
    """Get all available model keys."""
    return list(MODEL_TYPES.keys())


def get_model_directory_prefix(model_key: str) -> str:
    """Get the directory prefix for a model type."""
    return MODEL_DIRECTORY_MAPPING.get(model_key, model_key)


def get_model_status_key(model_key: str) -> str:
    """Get the status key used in system status checks."""
    return MODEL_STATUS_KEYS.get(model_key, model_key)


def get_training_stage_name(model_key: str) -> str:
    """Get the training stage name for a model type."""
    return TRAINING_STAGES.get(model_key, model_key)


def get_data_type(model_key: str) -> str:
    """Get the data type for a model type."""
    return DATA_TYPE_MAPPINGS.get(model_key, model_key)


def get_score_field_name(model_key: str) -> str:
    """Get the score field name for a model type."""
    return SCORE_FIELD_NAMES.get(model_key, f"{model_key}_score")


def get_display_name(model_key: str) -> str:
    """Get the display name for UI purposes."""
    return DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())


def get_model_loading_type(model_key: str) -> str:
    """Get the model loading type for pipeline initialization."""
    return MODEL_LOADING_TYPES.get(model_key, "auto")


def get_tier_info(model_key: str) -> Dict[str, Any]:
    """Get tier information for a model type."""
    config = get_model_config(model_key)
    return {
        "tier": config.get("tier", "unknown"),
        "purpose": config.get("purpose", "Unknown purpose"),
        "performance": config.get("performance", "Unknown performance"),
        "techniques": config.get("techniques", []),
    }


def validate_model_key(model_key: str) -> bool:
    """Validate if a model key exists in the configuration."""
    return model_key in MODEL_TYPES


def get_model_summary() -> Dict[str, Any]:
    """Get a summary of all model configurations."""
    summary = {}
    for key, config in MODEL_TYPES.items():
        summary[key] = {
            "name": config["name"],
            "tier": config["tier"],
            "purpose": config["purpose"],
            "performance": config["performance"],
            "techniques": config["techniques"],
        }
    return summary
