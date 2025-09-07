from pathlib import Path
from typing import Optional

import yaml
from .schemas import Config

# Import centralized paths configuration
from .paths import (
    TRAINING_DATA_PATHS,
    DATA_QUALITY_THRESHOLDS,
    validate_training_data_paths,
    get_training_data_path,
    ensure_real_data_available,
    get_data_source_info,
)

# Import centralized model configuration
from .models import (
    MODEL_TYPES,
    MODEL_DIRECTORY_MAPPING,
    MODEL_STATUS_KEYS,
    TRAINING_STAGES,
    DATA_TYPE_MAPPINGS,
    SCORE_FIELD_NAMES,
    DISPLAY_NAMES,
    MODEL_LOADING_TYPES,
    get_model_config,
    get_all_model_keys,
    get_model_directory_prefix,
    get_model_status_key,
    get_training_stage_name,
    get_data_type,
    get_score_field_name,
    get_display_name,
    get_model_loading_type,
    get_tier_info,
    validate_model_key,
    get_model_summary,
)

"""
Configuration Loader for LawBot
===============================

Loads, validates, and provides access to the application's configuration.
"""

_config: Optional[Config] = None


def _resolve_paths(config_data: dict, root_dir: Path) -> dict:
    """Resolve all relative paths in the config to absolute paths."""
    paths_config = config_data.get("paths", {})
    for key, value in paths_config.items():
        if isinstance(value, str):
            # Prepend root_dir, then resolve to handle '..' etc.
            paths_config[key] = str((root_dir / value).resolve())
    
    # Add the root directory itself to the paths config
    paths_config["root_dir"] = str(root_dir)
    
    # Add centralized training data paths
    paths_config["training_data"] = TRAINING_DATA_PATHS
    paths_config["data_quality"] = DATA_QUALITY_THRESHOLDS
    
    return config_data


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load, parse, and validate the configuration from a YAML file."""
    global _config
    if _config:
        return _config

    # Determine project root and default config path
    project_root = Path(__file__).parent.parent
    if config_path is None:
        config_path = project_root / "config" / "default.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Load the YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Resolve paths to be absolute
    config_data = _resolve_paths(config_data, project_root)

    # Validate with Pydantic
    try:
        _config = Config(**config_data)
        return _config
    except Exception as e:
        print(f"Configuration validation error: {e}")
        raise


# Provide a singleton instance of the configuration
config = load_config()

# Export centralized paths functions for easy access
__all__ = [
    "config",
    "TRAINING_DATA_PATHS",
    "DATA_QUALITY_THRESHOLDS", 
    "validate_training_data_paths",
    "get_training_data_path",
    "ensure_real_data_available",
    "get_data_source_info",
    # Model configuration exports
    "MODEL_TYPES",
    "MODEL_DIRECTORY_MAPPING",
    "MODEL_STATUS_KEYS",
    "TRAINING_STAGES",
    "DATA_TYPE_MAPPINGS",
    "SCORE_FIELD_NAMES",
    "DISPLAY_NAMES",
    "MODEL_LOADING_TYPES",
    "get_model_config",
    "get_all_model_keys",
    "get_model_directory_prefix",
    "get_model_status_key",
    "get_training_stage_name",
    "get_data_type",
    "get_score_field_name",
    "get_display_name",
    "get_model_loading_type",
    "get_tier_info",
    "validate_model_key",
    "get_model_summary",
]
