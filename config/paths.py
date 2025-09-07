"""
Centralized Path Configuration - LawBot v8.3
===========================================

Centralized configuration for all training data paths to ensure consistency
and prevent synthetic data fallback across all tiers.
"""

from pathlib import Path

def find_latest_processed_data_dir() -> Path:
    """Find the latest processed data directory with timestamp."""
    features_dir = Path("features")
    if not features_dir.exists():
        return Path("features/processed_data")  # fallback
    
    # Find all processed_data_* directories
    processed_dirs = [d for d in features_dir.iterdir() 
                     if d.is_dir() and d.name.startswith("processed_data_")]
    
    if not processed_dirs:
        return Path("features/processed_data")  # fallback
    
    # Sort by name (which includes timestamp) and get the latest
    latest_dir = sorted(processed_dirs)[-1]
    return latest_dir

# --- Training Data Paths (CRITICAL: Must exist for real training) ---
def get_training_data_paths():
    """Get training data paths with dynamic directory detection."""
    base_dir = find_latest_processed_data_dir()
    
    return {
        # Primary paths from run_preparation.py
        "primary": {
            "bi_encoder": base_dir / "bi_encoder_train.jsonl",
            "cross_encoder": base_dir / "cross_encoder_train.jsonl",
            "processed_corpus": base_dir / "processed_corpus.json",
            "training_data": base_dir / "training_data.jsonl",
            "negative_pool": base_dir / "negative_pool.jsonl",
        },
    }

TRAINING_DATA_PATHS = get_training_data_paths()

# Add alternative paths for flexibility
TRAINING_DATA_PATHS["alternative"] = {
        "bi_encoder": [
            Path("features/processed_data/bi_encoder_train.jsonl"),
            Path("features/bi_encoder_train.jsonl"),
            Path("data/processed/bi_encoder_train.jsonl"),
        ],
        "cross_encoder": [
            Path("features/processed_data/cross_encoder_train.jsonl"),
            Path("features/cross_encoder_train.jsonl"),
            Path("data/processed/cross_encoder_train.jsonl"),
        ],
        "training_data": [
            Path("features/processed_data/training_data.jsonl"),
            Path("features/training_data.jsonl"),
            Path("data/processed/training_data.jsonl"),
        ],
    }

# Add additional paths
TRAINING_DATA_PATHS["validation"] = {
    "tier_1": Path("features/validation_sets/tier_1_validation.jsonl"),
    "tier_2": Path("features/validation_sets/tier_2_validation.jsonl"),
    "tier_3": Path("features/validation_sets/tier_3_validation.jsonl"),
    "base_dir": Path("features/validation_sets"),
}

TRAINING_DATA_PATHS["models"] = {
    "bi_encoder": Path("models/bi_encoder"),
    "light_ranking": Path("models/light_ranking"),
    "cross_encoder": Path("models/cross_encoder"),
    "base_dir": Path("models"),
}

TRAINING_DATA_PATHS["checkpoints"] = Path("checkpoints")
TRAINING_DATA_PATHS["reports"] = Path("reports")
TRAINING_DATA_PATHS["logs"] = Path("logs")

# --- Data Quality Thresholds ---
DATA_QUALITY_THRESHOLDS = {
    "bi_encoder": {
        "min_triplets": 5,  # Minimum triplets for real training
        "min_queries": 3,  # Minimum unique queries
        "min_content_length": 50,  # Minimum content length
    },
    "light_ranking": {
        "min_examples": 5,  # Minimum training examples
        "min_queries": 3,  # Minimum unique queries
        "min_content_length": 50,  # Minimum content length
    },
    "cross_encoder": {
        "min_examples": 5,  # Minimum training examples
        "min_queries": 3,  # Minimum unique queries
        "min_content_length": 50,  # Minimum content length
    },
}


# --- Validation Functions ---
def validate_training_data_paths() -> dict:
    """
    Validate that all required training data paths exist.
    Automatically finds the latest processed data directory.

    Returns:
        dict: Validation results for each tier
    """
    validation_results = {
        "tier_1": {"status": "unknown", "data_source": "unknown", "file_count": 0},
        "tier_2": {"status": "unknown", "data_source": "unknown", "file_count": 0},
        "tier_3": {"status": "unknown", "data_source": "unknown", "file_count": 0},
        "overall": {"status": "unknown", "real_data_available": False},
    }

    # Find the latest processed data directory
    features_dir = Path("features")
    processed_dirs = [
        d
        for d in features_dir.iterdir()
        if d.is_dir() and d.name.startswith("processed_data_")
    ]

    if processed_dirs:
        # Sort by creation time and get the latest
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        validation_results["latest_data_dir"] = str(latest_dir)

        # Check Tier 1 (Bi-Encoder)
        tier1_files = [
            latest_dir / "bi_encoder_train.jsonl",
            latest_dir / "processed_corpus.json",
        ]
        tier1_existing = [f for f in tier1_files if f.exists()]

        if len(tier1_existing) >= 2:
            validation_results["tier_1"] = {
                "status": "ready",
                "data_source": "real",
                "file_count": len(tier1_existing),
                "files": [str(f) for f in tier1_existing],
            }
        else:
            validation_results["tier_1"] = {
                "status": "missing_data",
                "data_source": "missing",
                "file_count": len(tier1_existing),
                "missing_files": [str(f) for f in tier1_files if not f.exists()],
            }

        # Check Tier 2 (Light Ranking)
        tier2_files = [
            latest_dir / "training_data.jsonl",
            latest_dir / "negative_pool.jsonl",
        ]
        tier2_existing = [f for f in tier2_files if f.exists()]

        if len(tier2_existing) >= 2:
            validation_results["tier_2"] = {
                "status": "ready",
                "data_source": "real",
                "file_count": len(tier2_existing),
                "files": [str(f) for f in tier2_existing],
            }
        else:
            validation_results["tier_2"] = {
                "status": "missing_data",
                "data_source": "missing",
                "file_count": len(tier2_existing),
                "missing_files": [str(f) for f in tier2_files if not f.exists()],
            }

        # Check Tier 3 (Cross-Encoder)
        tier3_files = [
            latest_dir / "cross_encoder_train.jsonl",
            latest_dir / "processed_corpus.json",
        ]
        tier3_existing = [f for f in tier3_files if f.exists()]

        if len(tier3_existing) >= 2:
            validation_results["tier_3"] = {
                "status": "ready",
                "data_source": "real",
                "file_count": len(tier3_existing),
                "files": [str(f) for f in tier3_existing],
            }
        else:
            validation_results["tier_3"] = {
                "status": "missing_data",
                "data_source": "missing",
                "file_count": len(tier3_existing),
                "missing_files": [str(f) for f in tier3_files if not f.exists()],
            }
    else:
        # No processed data directories found
        validation_results["tier_1"] = {
            "status": "missing_data",
            "data_source": "missing",
            "file_count": 0,
            "missing_files": ["No processed_data_* directories found"],
        }
        validation_results["tier_2"] = {
            "status": "missing_data",
            "data_source": "missing",
            "file_count": 0,
            "missing_files": ["No processed_data_* directories found"],
        }
        validation_results["tier_3"] = {
            "status": "missing_data",
            "data_source": "missing",
            "file_count": 0,
            "missing_files": ["No processed_data_* directories found"],
        }

    # Overall status
    all_tiers_ready = all(
        validation_results[tier]["status"] == "ready"
        for tier in ["tier_1", "tier_2", "tier_3"]
    )

    validation_results["overall"] = {
        "status": "ready" if all_tiers_ready else "missing_data",
        "real_data_available": all_tiers_ready,
        "tiers_ready": sum(
            1
            for tier in ["tier_1", "tier_2", "tier_3"]
            if validation_results[tier]["status"] == "ready"
        ),
    }

    return validation_results


def get_training_data_path(tier: str, data_type: str) -> Path:
    """
    Get the appropriate training data path for a specific tier and data type.
    Automatically finds the latest processed data directory.

    Args:
        tier: Tier name ("tier_1", "tier_2", "tier_3")
        data_type: Type of data ("bi_encoder", "cross_encoder", "training_data", etc.)

    Returns:
        Path: Path to the training data file
    """
    # Find the latest processed data directory
    features_dir = Path("features")
    processed_dirs = [
        d
        for d in features_dir.iterdir()
        if d.is_dir() and d.name.startswith("processed_data_")
    ]

    if not processed_dirs:
        # Fallback to primary paths if no processed_data_* directories found
        if tier == "tier_1":
            if data_type == "bi_encoder":
                return TRAINING_DATA_PATHS["primary"]["bi_encoder"]
            elif data_type == "processed_corpus":
                return TRAINING_DATA_PATHS["primary"]["processed_corpus"]
        elif tier == "tier_2":
            if data_type == "training_data":
                return TRAINING_DATA_PATHS["primary"]["training_data"]
            elif data_type == "negative_pool":
                return TRAINING_DATA_PATHS["primary"]["negative_pool"]
        elif tier == "tier_3":
            if data_type == "cross_encoder":
                return TRAINING_DATA_PATHS["primary"]["cross_encoder"]
            elif data_type == "processed_corpus":
                return TRAINING_DATA_PATHS["primary"]["processed_corpus"]

        # Fallback to primary path
        return TRAINING_DATA_PATHS["primary"].get(data_type, Path(""))

    # Sort by creation time and get the latest
    latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)

    # Return paths from the latest directory
    if tier == "tier_1":
        if data_type == "bi_encoder":
            return latest_dir / "bi_encoder_train.jsonl"
        elif data_type == "processed_corpus":
            return latest_dir / "processed_corpus.json"
    elif tier == "tier_2":
        if data_type == "training_data":
            return latest_dir / "training_data.jsonl"
        elif data_type == "negative_pool":
            return latest_dir / "negative_pool.jsonl"
    elif tier == "tier_3":
        if data_type == "cross_encoder":
            return latest_dir / "cross_encoder_train.jsonl"
        elif data_type == "processed_corpus":
            return latest_dir / "processed_corpus.json"

    # Fallback to primary path
    return TRAINING_DATA_PATHS["primary"].get(data_type, Path(""))


def ensure_real_data_available(tier: str) -> bool:
    """
    Ensure that real training data is available for a specific tier.

    Args:
        tier: Tier name ("tier_1", "tier_2", "tier_3")

    Returns:
        bool: True if real data is available, False otherwise
    """
    validation = validate_training_data_paths()
    return validation[tier]["data_source"] == "real"


def get_data_source_info(tier: str) -> dict:
    """
    Get detailed information about data source for a specific tier.

    Args:
        tier: Tier name ("tier_1", "tier_2", "tier_3")

    Returns:
        dict: Data source information
    """
    validation = validate_training_data_paths()
    return validation[tier]
