import json
from pathlib import Path
from typing import List, Dict, Any, Union
import logging

"""
I/O Utilities - LawBot v8.0
============================

Utility functions for file I/O operations including JSON and JSONL handling.
"""

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: Union[Path, str]) -> Path:
    """Ensures a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object of the ensured directory
    """
    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory ensured: {directory_path}")
    return directory_path


def load_json(file_path: Union[Path, str]) -> Union[Dict, List]:
    """Loads a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json(data: Union[Dict, List], file_path: Union[Path, str], indent: int = 4):
    """Saves data to a standard JSON file.

    Args:
        data: Data to save.
        file_path: Path to save the file.
        indent: JSON indentation.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise


def load_jsonl(file_path: Union[Path, str]) -> List[Dict]:
    """Loads a JSONL file robustly, handling different line endings.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries loaded from the file

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If any line contains invalid JSON
    """
    data = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by newline character, which works for both LF and CRLF after reading
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            if line.strip():  # handle empty lines that might result from splitting
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num} in {file_path}: {e}")
                    raise

        logger.debug(f"Loaded {len(data)} items from {file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load JSONL from {file_path}: {e}")
        raise


def save_jsonl(data: List[Dict], file_path: Union[Path, str]):
    """Saves a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to save.
        file_path: Path to save the file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise
