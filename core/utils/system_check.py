#!/usr/bin/env python3
"""
System Check Utilities for LawBot
================================

Provides functions to check system status, model availability, and hardware information.
"""

import os
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from config.loader import config, MODEL_DIRECTORY_MAPPING, MODEL_STATUS_KEYS
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    # Fallback configuration
    config = None
    MODEL_DIRECTORY_MAPPING = {
        "bi_encoder": "bi-encoder",
        "light_reranker": "light-ranking",
        "cross_encoder": "combined-reranker-adapt",
    }
    MODEL_STATUS_KEYS = {
        "bi_encoder": "bi_encoder",
        "light_reranker": "light_reranker",
        "cross_encoder": "cross_encoder",
    }

from core.utils.logging_manager import get_logger

logger = get_logger(__name__)


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device and system information."""
    device_info = {
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "hardware": {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
            "memory_percent": psutil.virtual_memory().percent,
        },
        "gpu": {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
        },
    }

    # Check CUDA availability
    try:
        import torch

        device_info["gpu"]["cuda_available"] = torch.cuda.is_available()
        if device_info["gpu"]["cuda_available"]:
            device_info["gpu"]["cuda_version"] = torch.version.cuda
            device_info["gpu"]["gpu_count"] = torch.cuda.device_count()
            device_info["gpu"]["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        logger.info("PyTorch not available, skipping CUDA check")

    return device_info


def get_model_status() -> Dict[str, Dict[str, Any]]:
    """Checks for the existence, path, and size of all model types."""
    if not config:
        logger.warning("Config not available, using fallback model paths")
        models_dir = Path("models")
    else:
        models_dir = Path(config.paths.model_dir)

    status = {}

    # Use centralized model configuration
    # MODEL_DIRECTORY_MAPPING maps canonical model keys -> directory prefixes
    model_types = MODEL_DIRECTORY_MAPPING

    if not models_dir.exists():
        for key, dir_prefix in model_types.items():
            status[key] = {
                "status": "not_ready",
                "exists": False,
                "path": None,
                "size_mb": 0,
                "count": 0,
            }
        return status

    for key, dir_prefix in model_types.items():
        # Find directories that match the configured directory prefix
        model_dirs = [
            d
            for d in models_dir.iterdir()
            if d.is_dir() and d.name.startswith(dir_prefix)
        ]
        if model_dirs:
            latest_model = max(model_dirs, key=lambda p: p.stat().st_mtime)
            total_size = sum(
                f.stat().st_size for f in latest_model.rglob("*") if f.is_file()
            )
            status[key] = {
                "status": "ready",  # Add status key that system page expects
                "exists": True,
                "path": str(latest_model),
                "size_mb": total_size / (1024 * 1024),
                "count": len(model_dirs),
                "latest_name": latest_model.name,
                "directory_prefix": dir_prefix,
            }
        else:
            status[key] = {
                "status": "not_ready",  # Add status key that system page expects
                "exists": False,
                "path": None,
                "size_mb": 0,
                "count": 0,
                "directory_prefix": dir_prefix,
            }

    return status


def get_faiss_index_status() -> Dict[str, Any]:
    """Checks for FAISS index files."""
    if not config:
        logger.warning("Config not available, using fallback features path")
        features_dir = Path("features")
    else:
        features_dir = Path(config.paths.feature_dir)

    if not features_dir.exists():
        return {
            "status": "not_ready",  # Add status key that system page expects
            "ready": False,
            "exists": False,
            "files": [],
            "file_count": 0,
            "size_mb": 0,
            "index_size": 0,  # Add index_size for system page
            "document_count": 0,
        }

    # Check for FAISS index files in features directory
    faiss_files = []
    total_size = 0
    document_count = 0

    # Check main FAISS index
    faiss_index_path = features_dir / "faiss_index.bin"
    if faiss_index_path.exists():
        faiss_files.append("faiss_index.bin")
        total_size += faiss_index_path.stat().st_size

    # Check AID map
    aid_map_path = features_dir / "aid_map.json"
    if aid_map_path.exists():
        faiss_files.append("aid_map.json")
        total_size += aid_map_path.stat().st_size
        # Try to get document count from AID map
        try:
            import json

            with open(aid_map_path, "r", encoding="utf-8") as f:
                aid_data = json.load(f)
                document_count = len(aid_data)
        except:
            pass

    # Check index mapping
    index_to_aid_path = features_dir / "index_to_aid.json"
    if index_to_aid_path.exists():
        faiss_files.append("index_to_aid.json")
        total_size += index_to_aid_path.stat().st_size

    # Determine overall status
    has_required_files = (
        faiss_index_path.exists()
        and aid_map_path.exists()
        and index_to_aid_path.exists()
    )

    return {
        "status": "ready" if has_required_files else "not_ready",
        "ready": has_required_files,
        "exists": has_required_files,
        "files": faiss_files,
        "file_count": len(faiss_files),
        "size_mb": total_size / (1024 * 1024),
        "index_size": document_count,  # Use actual document count instead of file size
        "document_count": document_count,
    }


def get_directory_status() -> Dict[str, bool]:
    """Checks for the existence of key project directories."""
    dirs_to_check = ["data", "reports", "logs", "features", "training", "evaluation"]
    return {d: Path(d).exists() for d in dirs_to_check}


def load_latest_evaluation_report() -> Optional[Dict[str, Any]]:
    """Finds and loads the most recent evaluation JSON file from the 'reports' dir."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None

    all_reports = list(reports_dir.glob("*evaluation*.json"))
    if not all_reports:
        return None

    latest_report_path = max(all_reports, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest_report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return None


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status including models and FAISS index."""
    try:
        device_info = get_device_info()
        model_status = get_model_status()
        faiss_status = get_faiss_index_status()

        # Calculate overall system health
        models_ready = sum(1 for m in model_status.values() if m.get("exists", False))
        total_models = len(model_status)
        model_health = (models_ready / total_models) * 100 if total_models > 0 else 0

        system_health = {
            "overall": (
                "healthy"
                if model_health >= 75 and faiss_status["ready"]
                else "degraded" if model_health >= 50 else "unhealthy"
            ),
            "score": round(model_health, 1),
            "models_ready": models_ready,
            "total_models": total_models,
            "faiss_ready": faiss_status["ready"],
        }

        return {
            "device_info": device_info,
            "model_status": model_status,
            "faiss_status": faiss_status,
            "system_health": system_health,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "error": str(e),
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }


def check_model_availability(model_name: str) -> Dict[str, Any]:
    """Check availability of a specific model."""
    model_status = get_model_status()

    if model_name in model_status:
        return model_status[model_name]
    else:
        return {
            "status": "unknown",
            "exists": False,
            "path": None,
            "size_mb": 0,
            "count": 0,
        }


def get_available_models() -> List[str]:
    """Get list of available (ready) models."""
    model_status = get_model_status()
    return [
        name for name, status in model_status.items() if status.get("exists", False)
    ]


def get_missing_models() -> List[str]:
    """Get list of missing (not ready) models."""
    model_status = get_model_status()
    return [
        name for name, status in model_status.items() if not status.get("exists", False)
    ]


def get_dataset_status() -> Dict[str, Any]:
    """Get comprehensive dataset information and statistics with intelligent counting and caching."""
    try:
        # Import json module
        import json
        
        # Check if we have cached data (cache for 5 minutes)
        cache_file = Path("logs/dataset_status_cache.json")
        if cache_file.exists():
            import time
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 300:  # 5 minutes cache
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                        logger.info("Using cached dataset status")
                        return cached_data
                except Exception as e:
                    logger.warning(f"Failed to read cache: {e}")
        
        dataset_info = {
            "raw_data": {},
            "processed_data": {},
            "validation_sets": {},
            "overall_stats": {},
            "cached_at": __import__("datetime").datetime.now().isoformat()
        }
        
        # Check raw data with intelligent counting
        raw_data_dir = Path("data/raw")
        if raw_data_dir.exists():
            raw_files = list(raw_data_dir.glob("*.json"))
            dataset_info["raw_data"] = {
                "exists": True,
                "file_count": len(raw_files),
                "files": {},
                "total_articles": 0,
                "total_questions": 0
            }
            
            for file_path in raw_files:
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        # Intelligent counting based on file structure
                        if isinstance(data, list):
                            record_count = len(data)
                            
                            # Special handling for legal_corpus.json
                            if file_path.name == "legal_corpus.json":
                                # Count total articles across all laws
                                total_articles = 0
                                for law_item in data:
                                    if isinstance(law_item, dict) and "content" in law_item:
                                        total_articles += len(law_item["content"])
                                dataset_info["raw_data"]["total_articles"] = total_articles
                                record_count = total_articles  # Use total articles count
                            
                            # Special handling for train.json and public_test.json
                            elif file_path.name in ["train.json", "public_test.json"]:
                                dataset_info["raw_data"]["total_questions"] += record_count
                            
                        elif isinstance(data, dict):
                            if "content" in data:
                                record_count = len(data["content"])
                            else:
                                record_count = 1
                        else:
                            record_count = 0
                            
                        dataset_info["raw_data"]["files"][file_path.name] = {
                            "size_mb": round(file_size, 2),
                            "record_count": record_count,
                            "path": str(file_path),
                            "data_type": "list" if isinstance(data, list) else "dict" if isinstance(data, dict) else "unknown"
                        }
                        
                        # Add specific metadata for known file types
                        if file_path.name == "legal_corpus.json":
                            dataset_info["raw_data"]["files"][file_path.name]["law_count"] = len(data)
                            dataset_info["raw_data"]["files"][file_path.name]["article_count"] = record_count
                        elif file_path.name == "train.json":
                            dataset_info["raw_data"]["files"][file_path.name]["question_count"] = record_count
                        elif file_path.name == "public_test.json":
                            dataset_info["raw_data"]["files"][file_path.name]["question_count"] = record_count
                            
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")
                    dataset_info["raw_data"]["files"][file_path.name] = {
                        "size_mb": 0,
                        "record_count": 0,
                        "path": str(file_path),
                        "error": str(e)
                    }
        
        # Check processed data
        features_dir = Path("features")
        if features_dir.exists():
            # Find latest processed data directory
            processed_dirs = [
                d for d in features_dir.iterdir() 
                if d.is_dir() and d.name.startswith("processed_data_")
            ]
            
            if processed_dirs:
                latest_dir = max(processed_dirs, key=lambda p: p.stat().st_mtime)
                dataset_info["processed_data"] = {
                    "exists": True,
                    "latest_directory": latest_dir.name,
                    "files": {}
                }
                
                # Check metadata.json first
                metadata_path = latest_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            dataset_info["processed_data"]["metadata"] = metadata
                    except Exception as e:
                        logger.warning(f"Could not read metadata: {e}")
                
                # Check individual files
                for file_path in latest_dir.glob("*.jsonl"):
                    try:
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        # Count lines for JSONL files
                        with open(file_path, "r", encoding="utf-8") as f:
                            line_count = sum(1 for _ in f)
                        
                        dataset_info["processed_data"]["files"][file_path.name] = {
                            "size_mb": round(file_size, 2),
                            "record_count": line_count,
                            "path": str(file_path)
                        }
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
        
        # Check validation sets
        validation_dir = features_dir / "validation_sets"
        if validation_dir.exists():
            validation_files = list(validation_dir.glob("*.jsonl"))
            dataset_info["validation_sets"] = {
                "exists": True,
                "file_count": len(validation_files),
                "files": {},
                "total_records": 0,
                "empty_files": 0,
                "error_files": 0
            }
            
            for file_path in validation_files:
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    
                    # Count lines more robustly
                    line_count = 0
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:  # Only count non-empty lines
                                    line_count += 1
                    except Exception as read_error:
                        logger.warning(f"Could not read lines from {file_path}: {read_error}")
                        line_count = 0
                    
                    # Determine file status
                    if line_count == 0:
                        file_status = "empty"
                        dataset_info["validation_sets"]["empty_files"] += 1
                    else:
                        file_status = "ok"
                        dataset_info["validation_sets"]["total_records"] += line_count
                    
                    dataset_info["validation_sets"]["files"][file_path.name] = {
                        "size_mb": round(file_size, 2),
                        "record_count": line_count,
                        "path": str(file_path),
                        "status": file_status,
                        "is_empty": line_count == 0
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not process {file_path}: {e}")
                    dataset_info["validation_sets"]["error_files"] += 1
                    dataset_info["validation_sets"]["files"][file_path.name] = {
                        "size_mb": 0,
                        "record_count": 0,
                        "path": str(file_path),
                        "status": "error",
                        "error": str(e),
                        "is_empty": True
                    }
        else:
            dataset_info["validation_sets"] = {
                "exists": False,
                "file_count": 0,
                "files": {},
                "total_records": 0,
                "empty_files": 0,
                "error_files": 0
            }
        
        # Calculate overall stats with detailed breakdown
        total_files = (
            dataset_info["raw_data"].get("file_count", 0) +
            len(dataset_info["processed_data"].get("files", {})) +
            dataset_info["validation_sets"].get("file_count", 0)
        )
        
        total_size = sum(
            file_info["size_mb"] 
            for file_info in dataset_info["raw_data"].get("files", {}).values()
        ) + sum(
            file_info["size_mb"] 
            for file_info in dataset_info["processed_data"].get("files", {}).values()
        ) + sum(
            file_info["size_mb"] 
            for file_info in dataset_info["validation_sets"].get("files", {}).values()
        )
        
        # Calculate total records across all data types
        total_raw_records = sum(
            file_info["record_count"] 
            for file_info in dataset_info["raw_data"].get("files", {}).values()
            if "error" not in file_info
        )
        
        total_processed_records = sum(
            file_info["record_count"] 
            for file_info in dataset_info["processed_data"].get("files", {}).values()
        )
        
        # Use the pre-calculated total_records for validation sets
        total_validation_records = dataset_info["validation_sets"].get("total_records", 0)
        
        dataset_info["overall_stats"] = {
            "total_files": total_files,
            "total_size_mb": round(total_size, 2),
            "total_raw_records": total_raw_records,
            "total_processed_records": total_processed_records,
            "total_validation_records": total_validation_records,
            "total_all_records": total_raw_records + total_processed_records + total_validation_records,
            "data_available": total_files > 0
        }
        
        # Cache the results
        try:
            cache_file.parent.mkdir(exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to cache dataset status: {e}")
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"Error getting dataset status: {e}")
        return {
            "error": str(e),
            "raw_data": {"exists": False},
            "processed_data": {"exists": False},
            "validation_sets": {"exists": False},
            "overall_stats": {"total_files": 0, "total_size_mb": 0, "data_available": False}
        }


def clear_dataset_status_cache() -> bool:
    """Clear the dataset status cache to force refresh."""
    try:
        cache_file = Path("logs/dataset_status_cache.json")
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Dataset status cache cleared")
            return True
        else:
            logger.info("No dataset status cache to clear")
            return True
    except Exception as e:
        logger.error(f"Failed to clear dataset status cache: {e}")
        return False


def get_hardware_requirements() -> Dict[str, Any]:
    """Get hardware requirements and recommendations for the chatbot application."""
    requirements = {
        "minimum": {
            "cpu_cores": 2,
            "memory_gb": 4,
            "storage_gb": 10,
            "gpu": "Optional (CPU mode supported)"
        },
        "recommended": {
            "cpu_cores": 4,
            "memory_gb": 8,
            "storage_gb": 20,
            "gpu": "NVIDIA GPU with 4GB+ VRAM"
        },
        "optimal": {
            "cpu_cores": 8,
            "memory_gb": 16,
            "storage_gb": 50,
            "gpu": "NVIDIA GPU with 8GB+ VRAM"
        },
        "current_system": {},
        "compatibility": {}
    }
    
    try:
        # Get current system info
        device_info = get_device_info()
        requirements["current_system"] = {
            "cpu_cores": device_info["hardware"]["cpu_count"],
            "memory_gb": device_info["hardware"]["memory_total_gb"],
            "gpu_available": device_info["gpu"]["cuda_available"],
            "gpu_count": device_info["gpu"]["gpu_count"],
            "gpu_names": device_info["gpu"]["gpu_names"]
        }
        
        # Check compatibility
        current_cpu = device_info["hardware"]["cpu_count"]
        current_memory = device_info["hardware"]["memory_total_gb"]
        current_gpu = device_info["gpu"]["cuda_available"]
        
        requirements["compatibility"] = {
            "meets_minimum": (
                current_cpu >= requirements["minimum"]["cpu_cores"] and
                current_memory >= requirements["minimum"]["memory_gb"]
            ),
            "meets_recommended": (
                current_cpu >= requirements["recommended"]["cpu_cores"] and
                current_memory >= requirements["recommended"]["memory_gb"]
            ),
            "meets_optimal": (
                current_cpu >= requirements["optimal"]["cpu_cores"] and
                current_memory >= requirements["optimal"]["memory_gb"] and
                current_gpu
            ),
            "performance_level": "minimum"
        }
        
        # Determine performance level
        if requirements["compatibility"]["meets_optimal"]:
            requirements["compatibility"]["performance_level"] = "optimal"
        elif requirements["compatibility"]["meets_recommended"]:
            requirements["compatibility"]["performance_level"] = "recommended"
        elif requirements["compatibility"]["meets_minimum"]:
            requirements["compatibility"]["performance_level"] = "minimum"
        else:
            requirements["compatibility"]["performance_level"] = "below_minimum"
        
        return requirements
        
    except Exception as e:
        logger.error(f"Error getting hardware requirements: {e}")
        return {
            "error": str(e),
            "minimum": requirements["minimum"],
            "recommended": requirements["recommended"],
            "optimal": requirements["optimal"]
        }


def validate_system_requirements() -> Dict[str, Any]:
    """Validate if the system meets minimum requirements."""
    device_info = get_device_info()

    requirements = {
        "python_version": {
            "required": "3.8",
            "current": device_info["system"]["python_version"],
            "met": True,  # Will be updated below
        },
        "memory": {
            "required_gb": 4,
            "current_gb": device_info["hardware"]["memory_total_gb"],
            "met": True,  # Will be updated below
        },
        "cpu": {
            "required_cores": 2,
            "current_cores": device_info["hardware"]["cpu_count"],
            "met": True,  # Will be updated below
        },
    }

    # Check Python version
    import pkg_resources

    current_version = pkg_resources.parse_version(
        device_info["system"]["python_version"]
    )
    required_version = pkg_resources.parse_version(
        requirements["python_version"]["required"]
    )
    requirements["python_version"]["met"] = current_version >= required_version

    # Check memory
    requirements["memory"]["met"] = (
        device_info["hardware"]["memory_total_gb"]
        >= requirements["memory"]["required_gb"]
    )

    # Check CPU cores
    requirements["cpu"]["met"] = (
        device_info["hardware"]["cpu_count"] >= requirements["cpu"]["required_cores"]
    )

    # Overall validation
    all_requirements_met = all(req["met"] for req in requirements.values())

    return {
        "requirements": requirements,
        "all_met": all_requirements_met,
        "overall_status": "ready" if all_requirements_met else "requirements_not_met",
    }
