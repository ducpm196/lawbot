#!/usr/bin/env python3
"""
Report Manager - LawBot v8.1 (Optimized)
=========================================

Centralized report management for the LawBot system with optimized
report file management and reduced scattering.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging

from config.loader import config

# Global state to track report management
_reports_configured = False
_active_reports: Dict[str, str] = {}  # report_type -> latest_file_path
_workflow_timestamp: Optional[str] = None

logger = logging.getLogger(__name__)


def setup_report_management(workflow_timestamp: Optional[str] = None):
    """SỬA: Setup centralized report management.
    
    Args:
        workflow_timestamp: Optional timestamp for workflow reports to ensure consistency
    """
    global _reports_configured, _workflow_timestamp
    
    if workflow_timestamp:
        _workflow_timestamp = workflow_timestamp
    
    _reports_configured = True
    logger.info("✅ Report management configured")


def get_reports_directory() -> Path:
    """Get the reports directory path."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def save_report(
    report_type: str,
    data: Dict[str, Any],
    timestamp: Optional[str] = None,
    keep_latest_only: bool = True
) -> str:
    """SỬA: Save report with centralized management.
    
    Args:
        report_type: Type of report (e.g., 'comprehensive_evaluation', 'bi_encoder')
        data: Report data to save
        timestamp: Optional timestamp, uses current time if not provided
        keep_latest_only: If True, keeps only the latest report of this type
        
    Returns:
        Path to saved report file
    """
    global _workflow_timestamp
    
    if not _reports_configured:
        setup_report_management()
    
    reports_dir = get_reports_directory()
    
    # Use workflow timestamp if available, otherwise use provided or current timestamp
    if timestamp:
        report_timestamp = timestamp
    elif _workflow_timestamp and report_type in ['comprehensive_evaluation', 'workflow_summary']:
        report_timestamp = _workflow_timestamp
    else:
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename
    filename = f"{report_type}_{report_timestamp}.json"
    filepath = reports_dir / filename
    
    # Add metadata to report data
    enhanced_data = {
        "report_type": report_type,
        "timestamp": datetime.now().isoformat(),
        "report_timestamp": report_timestamp,
        "metadata": {
            "version": "v8.1",
            "generated_by": "report_manager",
            "workflow_timestamp": _workflow_timestamp,
        },
        "data": data
    }
    
    # Save report
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        # Update active reports
        _active_reports[report_type] = str(filepath)
        
        # Cleanup old reports if requested
        if keep_latest_only:
            cleanup_old_reports(report_type, keep_count=1)
        
        logger.info(f"✅ Report saved: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"❌ Failed to save report {report_type}: {e}")
        raise


def load_latest_report(report_type: str) -> Optional[Dict[str, Any]]:
    """Load the latest report of a specific type.
    
    Args:
        report_type: Type of report to load
        
    Returns:
        Report data or None if not found
    """
    reports_dir = get_reports_directory()
    
    # Find latest report of this type
    pattern = f"{report_type}_*.json"
    report_files = list(reports_dir.glob(pattern))
    
    if not report_files:
        return None
    
    # Get the most recent file
    latest_file = max(report_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load report {latest_file}: {e}")
        return None


def cleanup_old_reports(
    report_type: Optional[str] = None,
    keep_count: int = 3,
    max_age_days: int = 7
):
    """SỬA: Cleanup old reports to prevent accumulation.
    
    Args:
        report_type: Specific report type to cleanup, or None for all types
        keep_count: Number of latest reports to keep per type
        max_age_days: Maximum age in days before cleanup
    """
    reports_dir = get_reports_directory()
    if not reports_dir.exists():
        return
    
    # Get all report files
    if report_type:
        pattern = f"{report_type}_*.json"
        report_files = list(reports_dir.glob(pattern))
    else:
        report_files = list(reports_dir.glob("*_*.json"))
    
    if not report_files:
        return
    
    # Group by report type
    report_groups = {}
    for file in report_files:
        # Extract report type from filename
        name_parts = file.stem.split('_')
        if len(name_parts) >= 2:
            type_part = '_'.join(name_parts[:-1])  # Everything except timestamp
            if type_part not in report_groups:
                report_groups[type_part] = []
            report_groups[type_part].append(file)
    
    # Cleanup each group
    total_removed = 0
    for group_type, files in report_groups.items():
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        files_to_remove = files[keep_count:]
        
        # Also remove files older than max_age_days
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        for file in files:
            if file not in files_to_remove:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                if file_time < cutoff_time:
                    files_to_remove.append(file)
        
        # Remove duplicates
        files_to_remove = list(set(files_to_remove))
        
        # Delete files
        for file in files_to_remove:
            try:
                file.unlink()
                total_removed += 1
                logger.info(f"🗑️ Removed old report: {file.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove {file.name}: {e}")
    
    if total_removed > 0:
        logger.info(f"✅ Cleaned up {total_removed} old report files")


def get_report_summary() -> Dict[str, Any]:
    """Get summary of all reports.
    
    Returns:
        Dictionary with report summary information
    """
    reports_dir = get_reports_directory()
    if not reports_dir.exists():
        return {"error": "Reports directory not found"}
    
    # Get all report files
    report_files = list(reports_dir.glob("*_*.json"))
    
    # Group by type
    report_groups = {}
    total_size = 0
    
    for file in report_files:
        # Extract report type
        name_parts = file.stem.split('_')
        if len(name_parts) >= 2:
            type_part = '_'.join(name_parts[:-1])
            if type_part not in report_groups:
                report_groups[type_part] = []
            report_groups[type_part].append(file)
        
        total_size += file.stat().st_size
    
    # Create summary
    summary = {
        "total_reports": len(report_files),
        "total_size_mb": total_size / 1024 / 1024,
        "report_types": {},
        "active_reports": _active_reports,
        "workflow_timestamp": _workflow_timestamp
    }
    
    for report_type, files in report_groups.items():
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = files[0]
        
        summary["report_types"][report_type] = {
            "count": len(files),
            "latest_file": latest_file.name,
            "latest_size": latest_file.stat().st_size,
            "latest_modified": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        }
    
    return summary


def consolidate_reports(workflow_timestamp: str) -> str:
    """SỬA: Consolidate all reports for a workflow into a single summary.
    
    Args:
        workflow_timestamp: Workflow timestamp to consolidate
        
    Returns:
        Path to consolidated report
    """
    reports_dir = get_reports_directory()
    
    # Find all reports with this workflow timestamp
    pattern = f"*_{workflow_timestamp}.json"
    workflow_reports = list(reports_dir.glob(pattern))
    
    if not workflow_reports:
        logger.warning(f"No reports found for workflow {workflow_timestamp}")
        return None
    
    # Load all reports
    consolidated_data = {
        "workflow_timestamp": workflow_timestamp,
        "consolidated_at": datetime.now().isoformat(),
        "reports": {}
    }
    
    for report_file in workflow_reports:
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Extract report type from filename
            name_parts = report_file.stem.split('_')
            if len(name_parts) >= 2:
                report_type = '_'.join(name_parts[:-1])
                consolidated_data["reports"][report_type] = report_data
            
        except Exception as e:
            logger.error(f"Failed to load {report_file}: {e}")
    
    # Save consolidated report
    consolidated_filename = f"workflow_consolidated_{workflow_timestamp}.json"
    consolidated_path = reports_dir / consolidated_filename
    
    try:
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Consolidated report saved: {consolidated_path}")
        return str(consolidated_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to save consolidated report: {e}")
        return None


def reset_report_management():
    """SỬA: Reset report management state for testing."""
    global _reports_configured, _active_reports, _workflow_timestamp
    _reports_configured = False
    _active_reports.clear()
    _workflow_timestamp = None


def get_workflow_timestamp() -> Optional[str]:
    """Get the current workflow timestamp."""
    return _workflow_timestamp
