#!/usr/bin/env python3
"""
Progress Tracker - LawBot v8.0
==============================

Progress tracking and logging utilities for monitoring pipeline execution.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class ProgressTracker:
    """Progress tracker for monitoring pipeline execution steps."""

    def __init__(self, total_steps: int):
        """Initialize the progress tracker.

        Args:
            total_steps: Total number of steps to track
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.steps_info: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def start_step(
        self, step_name: str, step_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Start tracking a new step.

        Args:
            step_name: Name of the step
            step_info: Additional information about the step

        Returns:
            Start time of the step
        """
        self.current_step += 1
        step_start_time = time.time()

        if step_info is None:
            step_info = {}

        self.logger.info(
            f"[{self.current_step}/{self.total_steps}] Starting step: {step_name}..."
        )

        step_data = {
            "name": step_name,
            "info": step_info,
            "start_time": step_start_time,
            "end_time": None,
            "duration": None,
            "success": None,
        }

        self.steps_info.append(step_data)
        return step_start_time

    def end_step(self, step_start_time: float, success: bool):
        """End tracking the current step.

        Args:
            step_start_time: Start time returned by start_step
            success: Whether the step completed successfully
        """
        end_time = time.time()
        duration = end_time - step_start_time

        if self.steps_info:
            step_data = self.steps_info[-1]
            step_data.update(
                {"end_time": end_time, "duration": duration, "success": success}
            )

            status = "completed successfully" if success else "failed"
            self.logger.info(f"Step {step_data['name']} {status} in {duration:.2f}s.")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information.

        Returns:
            Dictionary containing progress information
        """
        completed_steps = sum(
            1 for step in self.steps_info if step.get("success") is not None
        )
        failed_steps = sum(
            1 for step in self.steps_info if step.get("success") is False
        )

        total_time = time.time() - self.start_time

        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "total_time": total_time,
            "progress_percentage": (
                (completed_steps / self.total_steps * 100)
                if self.total_steps > 0
                else 0
            ),
        }

    def get_summary(self) -> str:
        """Get a summary report of all steps.

        Returns:
            Formatted summary report string
        """
        total_time = time.time() - self.start_time
        return create_summary_report(self.steps_info, total_time)


class StepLogger:
    """Logger for individual pipeline steps."""

    def __init__(self, step_id: str):
        """Initialize the step logger.

        Args:
            step_id: Unique identifier for the step
        """
        self.step_id = step_id
        self.logger = logging.getLogger(f"StepLogger_{step_id}")

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(f"[STEP {self.step_id}] {message}")

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(f"[STEP {self.step_id}] {message}")

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(f"[STEP {self.step_id}] {message}")

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(f"[STEP {self.step_id}] {message}")

    def func_start(self, func_name: str):
        """Log the start of a function execution."""
        self.logger.info(f"[FUNC START] >> {func_name}")

    def func_end(self, func_name: str):
        """Log the end of a function execution."""
        self.logger.info(f"[FUNC END] << {func_name}")

    def step_progress(self, message: str):
        """Log step progress information."""
        self.logger.info(f"[PROGRESS] {message}")


def create_summary_report(steps_info: List[Dict[str, Any]], total_time: float) -> str:
    """Create a formatted summary report.

    Args:
        steps_info: List of step information dictionaries
        total_time: Total execution time

    Returns:
        Formatted summary report string
    """
    report = "\n" + "=" * 50 + "\n"
    report += "PIPELINE EXECUTION SUMMARY\n"
    report += "=" * 50 + "\n"

    for i, step in enumerate(steps_info, 1):
        status = (
            "✅ SUCCESS"
            if step.get("success")
            else "❌ FAILED" if step.get("success") is False else "⏳ RUNNING"
        )
        duration_str = (
            f"{step.get('duration', 0):.2f}s" if step.get("duration") else "N/A"
        )
        report += f"{i}. {step['name']:<40} {status:<10} ({duration_str})\n"

    report += "-" * 50 + "\n"
    report += f"Total Execution Time: {total_time:.2f}s\n"
    report += "=" * 50 + "\n"

    return report


class PipelineMonitor:
    """Advanced pipeline monitoring with real-time updates."""

    def __init__(self, pipeline_name: str):
        """Initialize the pipeline monitor.

        Args:
            pipeline_name: Name of the pipeline being monitored
        """
        self.pipeline_name = pipeline_name
        self.start_time = time.time()
        self.steps = []
        self.logger = logging.getLogger(f"PipelineMonitor_{pipeline_name}")

    def add_step(self, step_name: str, step_type: str = "default") -> str:
        """Add a new step to monitor.

        Args:
            step_name: Name of the step
            step_type: Type of the step

        Returns:
            Step ID for tracking
        """
        step_id = f"{step_name}_{len(self.steps)}"
        step_data = {
            "id": step_id,
            "name": step_name,
            "type": step_type,
            "start_time": None,
            "end_time": None,
            "duration": None,
            "status": "pending",
            "error": None,
        }
        self.steps.append(step_data)
        self.logger.info(f"Added step: {step_name} (ID: {step_id})")
        return step_id

    def start_step(self, step_id: str):
        """Mark a step as started.

        Args:
            step_id: ID of the step to start
        """
        for step in self.steps:
            if step["id"] == step_id:
                step["start_time"] = time.time()
                step["status"] = "running"
                self.logger.info(f"Started step: {step['name']}")
                break

    def complete_step(
        self, step_id: str, success: bool = True, error: Optional[str] = None
    ):
        """Mark a step as completed.

        Args:
            step_id: ID of the step to complete
            success: Whether the step completed successfully
            error: Error message if the step failed
        """
        for step in self.steps:
            if step["id"] == step_id:
                step["end_time"] = time.time()
                step["duration"] = (
                    step["end_time"] - step["start_time"]
                    if step["start_time"]
                    else None
                )
                step["status"] = "completed" if success else "failed"
                step["error"] = error

                status_msg = "completed successfully" if success else f"failed: {error}"
                self.logger.info(
                    f"Step {step['name']} {status_msg} in {step['duration']:.2f}s"
                )
                break

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Dictionary containing pipeline status information
        """
        total_steps = len(self.steps)
        completed_steps = sum(1 for step in self.steps if step["status"] == "completed")
        failed_steps = sum(1 for step in self.steps if step["status"] == "failed")
        running_steps = sum(1 for step in self.steps if step["status"] == "running")

        total_time = time.time() - self.start_time

        return {
            "pipeline_name": self.pipeline_name,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "running_steps": running_steps,
            "pending_steps": total_steps
            - completed_steps
            - failed_steps
            - running_steps,
            "total_time": total_time,
            "progress_percentage": (
                (completed_steps / total_steps * 100) if total_steps > 0 else 0
            ),
            "steps": self.steps,
        }
