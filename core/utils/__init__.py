"""
Core Utilities Module - LawBot v8.0
===================================

Core utility functions for LawBot including:
- I/O operations (JSON, JSONL)
- Logging management
- Versioning and path management
- Text canonicalization
"""

from .io import load_json, save_json, load_jsonl
from .logging_manager import (
    get_logger,
    setup_logging,
    cleanup_logs,
    log_pipeline_start,
    log_pipeline_end,
    log_step_start,
    log_step_end,
    log_error,
)
from .versioning import (
    get_timestamp,
    generate_versioned_path,
    get_latest_version_path,
    save_metadata,
)
from .canonicalization import canonicalize_aid

__all__ = [
    # I/O utilities
    "load_json",
    "save_json",
    "load_jsonl",
    # Logging utilities
    "get_logger",
    "setup_logging",
    "cleanup_logs",
    "log_pipeline_start",
    "log_pipeline_end",
    "log_step_start",
    "log_step_end",
    "log_error",
    # Versioning utilities
    "get_timestamp",
    "generate_versioned_path",
    "get_latest_version_path",
    "save_metadata",
    # Text utilities
    "canonicalize_aid",
]
