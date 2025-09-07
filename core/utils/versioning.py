from pathlib import Path
from datetime import datetime
import json
from typing import Optional

"""
Versioning Utilities - LawBot v8.0
==================================

Utility functions for managing versioned paths and timestamps.
"""


def get_timestamp() -> str:
    """Returns a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_versioned_path(base_dir: str, name: str) -> Path:
    """Generates a versioned directory path using a timestamp.

    This function only generates the Path object; it does not create the directory.

    Args:
        base_dir: The base directory (e.g., 'models', 'features').
        name: The name of the artifact (e.g., 'bi-encoder', 'processed_data').

    Returns:
        A Path object representing the versioned directory path.
    """
    timestamp = get_timestamp()
    return Path(base_dir) / f"{name}_{timestamp}"


def get_latest_version_path(base_dir: str, model_name: str) -> Optional[Path]:
    """Finds the path to the latest version of a model.

    Args:
        base_dir: Base directory path
        model_name: Name of the model

    Returns:
        Path to the latest version, or None if not found
    """
    paths = list(Path(base_dir).glob(f"{model_name}_*"))
    if not paths:
        return None
    return max(paths, key=lambda p: p.name)


def save_metadata(directory: Path, metadata: dict):
    """Saves metadata to a JSON file in a directory.

    Args:
        directory: Directory to save metadata in
        metadata: Metadata dictionary to save
    """
    # Ensure directory is a Path object
    if not isinstance(directory, Path):
        directory = Path(directory)

    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_file = directory / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
