import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Set
import os
import sys

from config.loader import config

"""
Logging Manager - LawBot v8.1 (Optimized)
==========================================

Centralized logging management for the LawBot system with optimized
log file management and reduced scattering.
"""

# Global state to track logging setup
_logging_configured = False
_active_loggers: Set[str] = set()
_workflow_timestamp: Optional[str] = None


def setup_logging(log_type: str = "app", workflow_timestamp: Optional[str] = None):
    """SỬA: Optimized logging setup with centralized management.
    
    Args:
        log_type: Type of logging ("app" for application, "workflow" for training pipeline)
        workflow_timestamp: Optional timestamp for workflow logging to ensure consistency
    """
    global _logging_configured, _active_loggers, _workflow_timestamp
    
    # SỬA: Prevent duplicate logging setup
    if _logging_configured and log_type in _active_loggers:
        return
    
    log_cfg = config.logging
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(exist_ok=True)

    if log_type == "workflow":
        # SỬA: Use global workflow timestamp to prevent multiple log files
        if workflow_timestamp:
            _workflow_timestamp = workflow_timestamp
            timestamp = workflow_timestamp
        elif _workflow_timestamp:
            timestamp = _workflow_timestamp
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _workflow_timestamp = timestamp

        log_file = log_dir / f"workflow_{timestamp}.log"
        logger_name = "LawBot.workflow"
    else:
        # App logging - daily rotation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"app_{timestamp[:8]}.log"  # YYYYMMDD format
        logger_name = "LawBot.app"

        # SỬA: Check if this is being called from run_app.py (not from app.py import)
        import inspect
        caller_frame = inspect.currentframe().f_back
        if caller_frame:
            caller_filename = caller_frame.f_code.co_filename
            if "app.py" in caller_filename:
                # Don't setup logging when app.py is imported
                return

    # SỬA: Centralized logging configuration
    if not _logging_configured:
        # Initial logging setup
        logging.basicConfig(
            level=getattr(logging, log_cfg.level.upper()),
            format=log_cfg.format,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),  # Console handler
                logging.FileHandler(log_file, encoding="utf-8"),  # File handler
            ],
        )
        _logging_configured = True
    else:
        # Add file handler for additional log type
        logger = logging.getLogger(logger_name)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                log_cfg.format,
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(file_handler)

    _active_loggers.add(log_type)
    
    # SỬA: Use get_logger instead of direct logging
    logger = get_logger(__name__)
    logger.info(f"✅ Logging configured for {log_type} -> {log_file}")
    logger.info(f"📊 Active loggers: {_active_loggers}")


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the specified name."""
    return logging.getLogger(name)


def cleanup_logs(max_workflow_logs: int = 3, max_app_logs: int = 7):
    """SỬA: Cleanup old log files to prevent accumulation.
    
    Args:
        max_workflow_logs: Maximum number of workflow log files to keep
        max_app_logs: Maximum number of app log files to keep
    """
    log_dir = Path(config.paths.log_dir)
    if not log_dir.exists():
        return
    
    # Cleanup workflow logs
    workflow_logs = sorted(
        [f for f in log_dir.glob("workflow_*.log")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if len(workflow_logs) > max_workflow_logs:
        for old_log in workflow_logs[max_workflow_logs:]:
            try:
                old_log.unlink()
                print(f"🗑️ Removed old workflow log: {old_log.name}")
            except Exception as e:
                print(f"⚠️ Failed to remove {old_log.name}: {e}")
    
    # Cleanup app logs
    app_logs = sorted(
        [f for f in log_dir.glob("app_*.log")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if len(app_logs) > max_app_logs:
        for old_log in app_logs[max_app_logs:]:
            try:
                old_log.unlink()
                print(f"🗑️ Removed old app log: {old_log.name}")
            except Exception as e:
                print(f"⚠️ Failed to remove {old_log.name}: {e}")


def get_workflow_timestamp() -> Optional[str]:
    """Get the current workflow timestamp."""
    return _workflow_timestamp


def reset_logging():
    """SỬA: Reset logging state for testing."""
    global _logging_configured, _active_loggers, _workflow_timestamp
    _logging_configured = False
    _active_loggers.clear()
    _workflow_timestamp = None
    
    # Clear all handlers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
    
    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


def log_system_info():
    """SỬA: Log system information for debugging."""
    logger = get_logger(__name__)
    
    logger.info("🔧 SYSTEM INFORMATION:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Working directory: {os.getcwd()}")
    logger.info(f"  Log directory: {config.paths.log_dir}")
    logger.info(f"  Active loggers: {_active_loggers}")
    logger.info(f"  Workflow timestamp: {_workflow_timestamp}")
    
    # Log file sizes
    log_dir = Path(config.paths.log_dir)
    if log_dir.exists():
        total_size = sum(f.stat().st_size for f in log_dir.glob("*.log"))
        logger.info(f"  Total log size: {total_size / 1024 / 1024:.2f} MB")


def cleanup_logs_advanced(
    max_age_hours: int = 24, max_size_mb: int = 100, max_workflow_logs: int = 5
):
    """Cleans up old log files based on age, total directory size, and workflow log count."""
    log_dir = Path(config.paths.log_dir)
    if not log_dir.exists():
        return

    try:
        current_time = datetime.now()
        log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        total_size = sum(p.stat().st_size for p in log_files) / (1024 * 1024)

        # Separate workflow and app logs
        workflow_logs = [f for f in log_files if f.name.startswith("workflow_")]
        app_logs = [f for f in log_files if f.name.startswith("app_")]

        # Keep only the latest workflow logs
        if len(workflow_logs) > max_workflow_logs:
            logs_to_remove = workflow_logs[:-max_workflow_logs]
            for log_file in logs_to_remove:
                try:
                    log_file.unlink()
                    get_logger(__name__).info(
                        f"Removed old workflow log: {log_file.name}"
                    )
                except Exception as e:
                    get_logger(__name__).warning(
                        f"Failed to remove {log_file.name}: {e}"
                    )

        # Clean up old app logs and large files
        for log_file in log_files:
            if log_file.exists():  # Check if still exists after workflow cleanup
                file_age_hr = (
                    current_time - datetime.fromtimestamp(log_file.stat().st_mtime)
                ).total_seconds() / 3600
                file_size_mb = log_file.stat().st_size / (1024 * 1024)

                if file_age_hr > max_age_hours or total_size > max_size_mb:
                    try:
                        log_file.unlink()
                        total_size -= file_size_mb
                        get_logger(__name__).info(
                            f"Removed old log file: {log_file.name}"
                        )
                    except Exception as e:
                        get_logger(__name__).warning(
                            f"Failed to remove {log_file.name}: {e}"
                        )

    except Exception as e:
        # Use print here as logging might be compromised
        print(f"[ERROR] Failed during log cleanup: {e}")


# --- Helper Functions for Consistent Logging ---


def log_pipeline_start(pipeline_name: str, stage: Optional[str] = None):
    """Logs the start of a pipeline or stage."""
    msg = f"🚀 Starting {pipeline_name}" + (f" - Stage: {stage}" if stage else "")
    get_logger("LawBot.workflow.pipeline").info(msg)


def log_pipeline_end(
    pipeline_name: str, stage: Optional[str] = None, success: bool = True
):
    """Logs the end of a pipeline or stage."""
    status = "✅ Completed" if success else "❌ Failed"
    msg = f"{status} {pipeline_name}" + (f" - Stage: {stage}" if stage else "")
    pipeline_logger = get_logger("LawBot.workflow.pipeline")
    if success:
        pipeline_logger.info(msg)
    else:
        pipeline_logger.error(msg)


def log_step_start(step_name: str):
    """Logs the start of a step."""
    get_logger("LawBot.workflow.step").info(f"📋 Starting step: {step_name}")


def log_step_end(
    step_name: str, success: bool = True, duration_s: Optional[float] = None
):
    """Logs the end of a step."""
    status = "✅ Completed" if success else "❌ Failed"
    duration_str = f" in {duration_s:.2f}s" if duration_s is not None else ""
    msg = f"{status} step: {step_name}{duration_str}"
    step_logger = get_logger("LawBot.workflow.step")
    if success:
        step_logger.info(msg)
    else:
        step_logger.error(msg)


def log_error(
    error: Exception, context: Optional[str] = None, logger_name: str = "LawBot.app"
):
    """Logs an error with optional context."""
    msg = f"Error in {context}: {error}" if context else str(error)
    get_logger(logger_name).error(msg, exc_info=True)
