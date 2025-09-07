#!/usr/bin/env python3
"""
Workflow Runner - LawBot v8.1 (Optimized)
=========================================

Main workflow orchestration script that manages the entire LawBot training pipeline.
Supports different presets, checkpoint management, stage validation, and smart CUDA handling.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging
import time
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.append(str(Path(__file__).parent))

try:
    from core.utils.versioning import get_latest_version_path, get_timestamp
    from core.utils.logging_manager import (
        get_logger,
        setup_logging,
        log_pipeline_start,
        log_pipeline_end,
        log_step_start,
        log_step_end,
    )
    from config.loader import config
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Please ensure all dependencies are installed and the project structure is correct."
    )
    sys.exit(1)

# --- Stage Definitions ---

STAGES = {
    "data_preparation": {
        "script": "data_processing/run_preparation.py",
        "description": "Prepare and preprocess training data",
        "dependencies": [],
        "checkpoint_file": "checkpoints/data_preparation.json",
        "required": True,
    },
    "bi_encoder": {
        "script": "training/run_bi_encoder.py",
        "description": "Bi-Encoder training for retrieval",
        "dependencies": ["data_preparation"],
        "checkpoint_file": "checkpoints/bi_encoder_training.json",
        "required": True,
    },
    "light_ranking": {
        "script": "training/run_light_ranking.py",
        "description": "Light ranking model training (Tier 2: Independent)",
        "dependencies": ["data_preparation", "bi_encoder"],
        "checkpoint_file": "checkpoints/light_ranking_training.json",
        "required": True,
    },
    "cross_encoder": {
        "script": "training/run_reranker.py",
        "description": "Cross-encoder reranker training (Tier 3: Inherits from Tier 2)",
        "dependencies": ["data_preparation", "bi_encoder", "light_ranking"],
        "checkpoint_file": "checkpoints/cross_encoder_training.json",
        "required": True,
    },
    "faiss_index": {
        "script": "training/run_create_faiss_index.py",
        "description": "Create FAISS index for retrieval using trained Bi-Encoder",
        "dependencies": ["data_preparation", "bi_encoder"],
        "checkpoint_file": "checkpoints/faiss_index_creation.json",
        "required": True,
    },
    "evaluation": {
        "script": "evaluation/run_evaluation.py",
        "description": "Run comprehensive evaluation for all 3 tiers",
        "dependencies": [
            "data_preparation",
            "bi_encoder",
            "light_ranking",
            "cross_encoder",
            "faiss_index",
        ],
        "checkpoint_file": "checkpoints/evaluation_completion.json",
        "required": True,
    },
}

# --- Preset Definitions ---

PRESETS = {
    "full": [
        "data_preparation",
        "bi_encoder",
        "light_ranking",
        "cross_encoder",
        "faiss_index",
        "evaluation",
    ],
    "training": ["data_preparation", "bi_encoder", "light_ranking", "cross_encoder"],
    "retrieval": ["data_preparation", "bi_encoder", "faiss_index"],
    "reranking": ["data_preparation", "light_ranking", "cross_encoder"],
    "evaluation_only": ["evaluation"],
    "data_only": ["data_preparation"],
}


class WorkflowCheckpoint:
    """Manages workflow checkpoints for resuming interrupted runs."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, stage: str, status: str, metadata: Dict[str, Any]):
        """Save a checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"
        checkpoint_data = {
            "stage": stage,
            "status": status,
            "timestamp": get_timestamp(),
            "metadata": metadata,
        }

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint for {stage}: {e}")
        return None

    def clear_checkpoint(self, stage: str):
        """Clear checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()


class WorkflowRunner:
    """Manages the execution of workflow stages with smart CUDA handling."""

    def __init__(self, stages: List[str], force_restart: bool = False):
        """Initialize workflow runner."""
        self.stages = stages
        self.force_restart = force_restart
        self.checkpoint_manager = WorkflowCheckpoint()
        self.logger = get_logger(__name__)

        # Smart CUDA detection and handling
        self.device_info = self._detect_device()
        self._setup_environment()

        # Progress tracking
        self.start_time = time.time()
        self.completed_stages = []
        self.failed_stages = []

    def _detect_device(self) -> Dict[str, Any]:
        """Detect available devices and CUDA status."""
        device_info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "device": "cpu",
            "recommended_device": "cpu",
        }

        try:
            import torch

            if torch.cuda.is_available():
                device_info["cuda_available"] = True
                device_info["cuda_version"] = torch.version.cuda
                device_info["gpu_count"] = torch.cuda.device_count()
                device_info["device"] = "cuda"
                device_info["recommended_device"] = "cuda"

                # Check CUDA memory
                try:
                    gpu_memory = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )
                    device_info["gpu_memory_gb"] = round(gpu_memory, 2)
                    if gpu_memory < 4:  # Less than 4GB
                        device_info["recommended_device"] = "cpu"
                        self.logger.warning(
                            f"GPU memory ({gpu_memory:.1f}GB) may be insufficient, recommending CPU"
                        )
                except:
                    pass

                self.logger.info(
                    f"✅ CUDA detected: {device_info['cuda_version']}, {device_info['gpu_count']} GPU(s)"
                )
            else:
                self.logger.info("ℹ️ CUDA not available, using CPU")

        except ImportError:
            self.logger.warning("PyTorch not available, cannot detect CUDA")

        return device_info

    def _setup_environment(self):
        """Setup environment variables based on device detection."""
        # Don't modify environment variables to ensure consistency with manual execution
        if self.device_info["recommended_device"] == "cpu":
            self.logger.info("🔧 Environment: CPU mode (no environment changes)")
        else:
            self.logger.info("🔧 Environment: CUDA mode (no environment changes)")

    def _run_with_device_fallback(self, script_path: str, stage_name: str) -> bool:
        """Run script with direct import to ensure consistency with manual execution."""
        try:
            self.logger.info(
                f"🚀 Running {stage_name} with direct import (consistent with manual execution)"
            )

            # Use direct import for ALL stages to ensure consistency with manual execution
            return self._run_with_direct_import(script_path, stage_name)

        except Exception as e:
            self.logger.error(f"❌ {stage_name} failed with exception: {e}")
            return False

    def _run_with_direct_import(self, script_path: str, stage_name: str) -> bool:
        """Run script by importing and executing directly to ensure consistency with manual execution."""
        try:
            self.logger.info(f"🚀 Running {stage_name} with direct import execution")
            self.logger.info(f"📁 Script: {script_path}")
            self.logger.info(
                f"🔧 Method: Direct import (identical to manual execution)"
            )
            self.logger.info(f"📂 Working directory: {Path.cwd()}")
            self.logger.info(
                f"🔍 Environment: No modifications (preserves manual execution context)"
            )

            # Import the module
            script_module = script_path.replace("/", ".").replace(".py", "")
            if script_module.startswith("training."):
                script_module = script_module
            elif script_module.startswith("data_processing."):
                script_module = script_module
            elif script_module.startswith("evaluation."):
                script_module = script_module

            # Execute the main function
            if "faiss_index" in stage_name:
                from training.run_create_faiss_index import create_faiss_index

                success = create_faiss_index()
            elif "bi_encoder" in stage_name:
                from training.run_bi_encoder import main as train_bi_encoder

                success = train_bi_encoder()
            elif "light_ranking" in stage_name:
                from training.run_light_ranking import main as train_light_ranking

                success = train_light_ranking()
            elif "cross_encoder" in stage_name:
                from training.run_reranker import main as train_cross_encoder

                success = train_cross_encoder()
            elif "data_preparation" in stage_name:
                from data_processing.run_preparation import run_data_preparation

                success = run_data_preparation()
            elif "evaluation" in stage_name:
                from evaluation.run_evaluation import main as run_evaluation

                success = run_evaluation()
            else:
                self.logger.warning(f"⚠️ Direct import not supported for {stage_name}")
                return False

            if success:
                self.logger.info(
                    f"✅ {stage_name} completed successfully with direct import"
                )
                return True
            else:
                self.logger.warning(f"⚠️ {stage_name} failed with direct import")
                return False

        except Exception as e:
            self.logger.error(
                f"❌ {stage_name} failed with direct import exception: {e}"
            )
            return False

    def validate_stages(self) -> bool:
        """Validate that all requested stages exist and dependencies are met."""
        for stage in self.stages:
            if stage not in STAGES:
                self.logger.error(f"Unknown stage: {stage}")
                return False

            # Check dependencies
            dependencies = STAGES[stage]["dependencies"]
            for dep in dependencies:
                if dep not in self.stages:
                    self.logger.error(
                        f"Stage {stage} requires {dep}, but it's not in the pipeline"
                    )
                    return False

        # CRITICAL: Validate data preparation requirement for training stages
        training_stages = ["bi_encoder", "light_ranking", "cross_encoder"]
        if any(stage in self.stages for stage in training_stages):
            if "data_preparation" not in self.stages:
                self.logger.error(
                    "❌ CRITICAL: Training stages require data_preparation for real data"
                )
                self.logger.error(
                    "💡 Without data_preparation, training cannot proceed (real data required)"
                )
                self.logger.info(
                    "🔧 Recommendation: Add data_preparation to your stages or use --preset full"
                )
                return False
            else:
                self.logger.info(
                    "✅ Data preparation stage included - real training data will be used"
                )

        # Validate model inheritance chain for 3-tier architecture
        if "cross_encoder" in self.stages and "light_ranking" not in self.stages:
            self.logger.error(
                "❌ Cross-Encoder (Tier 3) requires Light Ranking (Tier 2) for model inheritance"
            )
            return False

        if "faiss_index" in self.stages and "bi_encoder" not in self.stages:
            self.logger.error(
                "❌ FAISS Index requires Bi-Encoder (Tier 1) for retrieval"
            )
            return False

        return True

    def run_stage(self, stage: str) -> bool:
        """Run a single workflow stage with comprehensive reporting."""
        if stage not in STAGES:
            self.logger.error(f"❌ Unknown stage: {stage}")
            return False

        stage_config = STAGES[stage]
        script_path = stage_config["script"]
        description = stage_config["description"]

        # Check dependencies
        if not self._check_dependencies(stage):
            self.logger.error(f"❌ Dependencies not met for {stage}")
            return False

        # Check if already completed (with data freshness validation for data_preparation)
        if not self.force_restart and self.checkpoint_manager.load_checkpoint(stage):
            # Special handling for data_preparation: check data freshness
            if stage == "data_preparation":
                try:
                    from config.paths import validate_training_data_paths

                    validation_results = validate_training_data_paths()
                    overall_status = validation_results["overall"]

                    if overall_status["real_data_available"]:
                        self.logger.info(
                            f"✅ Stage {stage} already completed with fresh data, skipping"
                        )
                        self.completed_stages.append(stage)
                        return True
                    else:
                        self.logger.warning(
                            f"⚠️ Stage {stage} completed but data is stale/missing, re-running..."
                        )
                        # Continue to re-run the stage
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Could not validate data freshness: {e}, re-running {stage}"
                    )
                    # Continue to re-run the stage
            else:
                # For non-data stages, just skip if checkpoint exists
                self.logger.info(f"✅ Stage {stage} already completed, skipping")
                self.completed_stages.append(stage)
                return True

        # Start stage
        log_step_start(f"{stage}: {description}")
        stage_start_time = time.time()

        self.logger.info(f"🚀 Starting stage: {stage} - {description}")
        self.logger.info(f"📁 Script: {script_path}")
        self.logger.info(f"🔧 Device: {self.device_info['recommended_device'].upper()}")

        try:
            # Run stage with device fallback
            success = self._run_with_device_fallback(script_path, stage)

            # Calculate duration
            stage_duration = time.time() - stage_start_time

            if success:
                # Save checkpoint
                checkpoint_data = {
                    "stage": stage,
                    "status": "completed",
                    "device_used": self.device_info["recommended_device"],
                    "duration_seconds": stage_duration,
                    "timestamp": datetime.now().isoformat(),
                    "device_info": self.device_info,
                }
                self.checkpoint_manager.save_checkpoint(
                    stage, "completed", checkpoint_data
                )

                # Log success
                log_step_end(
                    f"{stage}: {description}", success=True, duration_s=stage_duration
                )
                self.logger.info(
                    f"✅ Stage {stage} completed successfully in {stage_duration:.2f}s"
                )

                # Add to completed stages
                self.completed_stages.append(stage)

                # Generate stage report
                self._generate_stage_report(stage, checkpoint_data)

                return True
            else:
                # Log failure
                log_step_end(
                    f"{stage}: {description}", success=False, duration_s=stage_duration
                )
                self.logger.error(
                    f"❌ Stage {stage} failed after {stage_duration:.2f}s"
                )

                # Add to failed stages
                self.failed_stages.append(stage)

                # Generate failure report
                self._generate_failure_report(stage, stage_duration)

                return False

        except Exception as e:
            stage_duration = time.time() - stage_start_time
            log_step_end(
                f"{stage}: {description}", success=False, duration_s=stage_duration
            )
            self.logger.error(f"❌ Stage {stage} failed with exception: {e}")
            self.failed_stages.append(stage)
            return False

    def _check_dependencies(self, stage: str) -> bool:
        """Check if all dependencies for a stage are met."""
        dependencies = STAGES[stage]["dependencies"]

        for dep in dependencies:
            if dep not in self.completed_stages:
                self.logger.warning(f"⚠️ Dependency {dep} not completed for {stage}")
                return False

        # CRITICAL: Enhanced data freshness validation for training stages
        if (
            stage in ["bi_encoder", "light_ranking", "cross_encoder"]
            and "data_preparation" in self.completed_stages
        ):
            # Use centralized paths config for validation
            try:
                from config.paths import (
                    validate_training_data_paths,
                    get_data_source_info,
                )

                # Validate all training data paths
                validation_results = validate_training_data_paths()
                self.logger.info("📊 Training data validation results:")

                for tier_name, tier_info in validation_results.items():
                    if tier_name != "overall" and isinstance(tier_info, dict):
                        status = tier_info.get("status", "unknown")
                        data_source = tier_info.get("data_source", "unknown")
                        file_count = tier_info.get("file_count", 0)

                        if status == "ready":
                            self.logger.info(
                                f"  ✅ {tier_name}: {status} ({data_source}, {file_count} files)"
                            )
                        else:
                            self.logger.warning(
                                f"  ⚠️ {tier_name}: {status} ({data_source}, {file_count} files)"
                            )

                # Check overall status
                overall_status = validation_results.get("overall", {})
                if not overall_status.get("real_data_available", False):
                    self.logger.error(
                        "❌ CRITICAL: Not all tiers have real training data available"
                    )
                    self.logger.error(
                        "💡 This will cause poor performance across all tiers"
                    )
                    self.logger.info(
                        "🔧 Recommendation: Re-run data_preparation stage first"
                    )
                    return False
                else:
                    self.logger.info("✅ All tiers have real training data available")

                # Tier-specific validation
                if stage == "bi_encoder":
                    tier1_info = get_data_source_info("tier_1")
                    if tier1_info.get("data_source") != "real":
                        self.logger.error(
                            "❌ CRITICAL: Tier 1 (Bi-Encoder) missing real training data"
                        )
                        return False

                elif stage == "light_ranking":
                    tier2_info = get_data_source_info("tier_2")
                    if tier2_info.get("data_source") != "real":
                        self.logger.error(
                            "❌ CRITICAL: Tier 2 (Light Ranking) missing real training data"
                        )
                        return False

                elif stage == "cross_encoder":
                    tier3_info = get_data_source_info("tier_3")
                    if tier3_info.get("data_source") != "real":
                        self.logger.error(
                            "❌ CRITICAL: Tier 3 (Cross-Encoder) missing real training data"
                        )
                        return False

            except ImportError as e:
                self.logger.warning(f"⚠️ Could not import centralized paths config: {e}")
                # Fallback to old validation method
                self._fallback_data_validation(stage)
            except Exception as e:
                self.logger.warning(f"⚠️ Centralized validation failed: {e}")
                # Fallback to old validation method
                self._fallback_data_validation(stage)

        return True

    def _fallback_data_validation(self, stage: str):
        """Fallback data validation method for backward compatibility."""
        try:
            # Check if data is fresh and sufficient
            data_checkpoint = self.checkpoint_manager.load_checkpoint(
                "data_preparation"
            )
            if data_checkpoint:
                data_timestamp = data_checkpoint.get("timestamp", "")
                if data_timestamp:
                    try:
                        from datetime import datetime

                        data_time = datetime.fromisoformat(
                            data_timestamp.replace("Z", "+00:00")
                        )
                        current_time = datetime.now().replace(tzinfo=data_time.tzinfo)
                        hours_old = (current_time - data_time).total_seconds() / 3600

                        if hours_old > 24:
                            self.logger.warning(
                                f"⚠️ Data is {hours_old:.1f} hours old. Consider re-running data_preparation"
                            )
                            self.logger.info(
                                "💡 Tip: Use --force-restart to ensure fresh data processing"
                            )

                        # CRITICAL: Check if real training data exists
                        training_data_paths = [
                            Path("features/processed_data/bi_encoder_train.jsonl"),
                            Path("features/processed_data/cross_encoder_train.jsonl"),
                            Path("features/processed_data/processed_corpus.json"),
                        ]

                        real_data_exists = any(
                            path.exists() for path in training_data_paths
                        )
                        if not real_data_exists:
                            self.logger.error(
                                "❌ CRITICAL: No real training data found from data_preparation.py"
                            )
                            self.logger.error(
                                "💡 This will cause poor performance across all tiers"
                            )
                            self.logger.info(
                                "🔧 Recommendation: Re-run data_preparation stage first"
                            )
                            return False
                        else:
                            self.logger.info(
                                "✅ Real training data found from data_preparation.py"
                            )

                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not validate data freshness: {e}")

        except Exception as e:
            self.logger.warning(f"⚠️ Fallback validation failed: {e}")

    def _generate_stage_report(self, stage: str, checkpoint_data: Dict[str, Any]):
        """SỬA: Generate detailed report for completed stage using centralized management."""
        try:
            from core.utils.report_manager import save_report, setup_report_management
            
            # Setup report management with workflow timestamp
            setup_report_management(self.workflow_timestamp)

            report_data = {
                "stage": stage,
                "status": "completed",
                "checkpoint_data": checkpoint_data,
                "device_info": self.device_info,
                "stage_config": STAGES[stage],
                "workflow_progress": {
                    "completed_stages": self.completed_stages,
                    "failed_stages": self.failed_stages,
                    "total_stages": len(self.stages),
                    "progress_percentage": len(self.completed_stages)
                    / len(self.stages)
                    * 100,
                },
            }

            # SỬA: Use centralized report management
            report_path = save_report(
                report_type=f"{stage}_report",
                data=report_data,
                timestamp=self.workflow_timestamp,
                keep_latest_only=True
            )

            self.logger.info(f"📊 Stage report saved: {report_path}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate stage report: {e}")

    def _generate_failure_report(self, stage: str, duration: float):
        """Generate failure report for failed stage."""
        try:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            timestamp = get_timestamp()
            report_path = reports_dir / f"{stage}_failure_{timestamp}.json"

            report = {
                "stage": stage,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "device_info": self.device_info,
                "stage_config": STAGES[stage],
                "error_analysis": {
                    "possible_causes": [
                        "CUDA memory issues",
                        "Model loading failures",
                        "Data processing errors",
                        "Timeout issues",
                    ],
                    "recommendations": [
                        "Check GPU memory availability",
                        "Verify data integrity",
                        "Review model configurations",
                        "Consider using CPU mode",
                    ],
                },
            }

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📊 Failure report saved: {report_path}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate failure report: {e}")

    def run_workflow(self) -> bool:
        """Run the complete workflow with comprehensive reporting."""
        log_pipeline_start("LawBot Training Pipeline")

        self.logger.info("🚀 Starting LawBot Training Pipeline")
        self.logger.info(f"📋 Total stages: {len(self.stages)}")
        self.logger.info(f"🔧 Device: {self.device_info['recommended_device'].upper()}")
        self.logger.info(f"📊 CUDA Info: {self.device_info}")

        # Validate stages
        if not self.validate_stages():
            self.logger.error("❌ Stage validation failed")
            return False

        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Run each stage
        for i, stage in enumerate(self.stages, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎯 Stage {i}/{len(self.stages)}: {stage}")
            self.logger.info(f"{'='*60}")

            if not self.run_stage(stage):
                self.logger.error(f"❌ Workflow failed at stage: {stage}")
                self._generate_workflow_summary()
                return False

            # Progress update
            progress = (i / len(self.stages)) * 100
            self.logger.info(
                f"📊 Progress: {progress:.1f}% ({i}/{len(self.stages)} stages completed)"
            )

        # Generate final summary
        self._generate_workflow_summary()

        # Log completion
        total_duration = time.time() - self.start_time
        log_pipeline_end("LawBot Training Pipeline", success=True)

        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎉 LAWBOT TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(
            f"⏱️ Total duration: {total_duration:.2f}s ({total_duration/3600:.2f}h)"
        )
        self.logger.info(f"✅ Completed stages: {len(self.completed_stages)}")
        self.logger.info(f"❌ Failed stages: {len(self.failed_stages)}")
        self.logger.info(f"{'='*60}")

        return True

    def _generate_workflow_summary(self):
        """Generate comprehensive workflow summary report."""
        try:
            timestamp = get_timestamp()
            summary_path = Path("reports") / f"workflow_summary_{timestamp}.json"

            total_duration = time.time() - self.start_time

            summary = {
                "workflow_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_duration_seconds": total_duration,
                    "total_duration_hours": total_duration / 3600,
                    "stages_requested": self.stages,
                    "preset_used": self._get_preset_name(),
                },
                "execution_summary": {
                    "completed_stages": self.completed_stages,
                    "failed_stages": self.failed_stages,
                    "total_stages": len(self.stages),
                    "success_rate": (
                        len(self.completed_stages) / len(self.stages) * 100
                        if self.stages
                        else 0
                    ),
                },
                "device_information": self.device_info,
                "stage_details": {},
                "recommendations": self._generate_recommendations(),
                "next_steps": self._generate_next_steps(),
            }

            # Add details for each stage
            for stage in self.stages:
                checkpoint = self.checkpoint_manager.load_checkpoint(stage)
                if checkpoint:
                    summary["stage_details"][stage] = checkpoint
                else:
                    summary["stage_details"][stage] = {"status": "not_started"}

            # Save summary
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📊 Workflow summary saved: {summary_path}")

            # Create human-readable summary
            self._create_human_readable_summary(summary, timestamp)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate workflow summary: {e}")

    def _get_preset_name(self) -> str:
        """Get the name of the preset being used."""
        for preset_name, preset_stages in PRESETS.items():
            if preset_stages == self.stages:
                return preset_name
        return "custom"

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []

        if self.failed_stages:
            recommendations.append("Review failed stages and address underlying issues")
            recommendations.append("Check system resources and device availability")

        if self.device_info["recommended_device"] == "cpu":
            recommendations.append("Consider upgrading GPU for faster training")

        if len(self.completed_stages) == len(self.stages):
            recommendations.append(
                "All stages completed successfully - ready for production use"
            )
            recommendations.append("Run evaluation to assess model performance")

        return recommendations

    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on execution results."""
        next_steps = []

        if "evaluation" in self.completed_stages:
            next_steps.append("Review evaluation results and model performance")
            next_steps.append("Deploy models to production environment")
            next_steps.append("Monitor model performance in real-world usage")
        elif "evaluation" in self.stages:
            next_steps.append("Complete evaluation stage to assess model performance")

        if "faiss_index" in self.completed_stages:
            next_steps.append("Test retrieval performance with sample queries")

        next_steps.append("Document model versions and configurations")
        next_steps.append("Setup monitoring and alerting for production models")

        return next_steps

    def _create_human_readable_summary(self, summary: Dict[str, Any], timestamp: str):
        """Create a human-readable summary file."""
        try:
            summary_path = Path("reports") / f"workflow_summary_{timestamp}.md"

            content = f"""# LawBot Training Pipeline - Workflow Summary

## 📊 Execution Overview
- **Status**: {'✅ SUCCESS' if summary['execution_summary']['success_rate'] == 100 else '⚠️ PARTIAL SUCCESS' if summary['execution_summary']['success_rate'] > 0 else '❌ FAILED'}
- **Total Duration**: {summary['workflow_info']['total_duration_hours']:.2f} hours
- **Stages Requested**: {len(summary['workflow_info']['stages_requested'])}
- **Stages Completed**: {len(summary['execution_summary']['completed_stages'])}
- **Success Rate**: {summary['execution_summary']['success_rate']:.1f}%

## 🎯 Stages Status
"""

            for stage in summary["workflow_info"]["stages_requested"]:
                stage_detail = summary["stage_details"].get(stage, {})
                status = stage_detail.get("status", "unknown")
                duration = stage_detail.get("duration_seconds", 0)

                if status == "completed":
                    content += f"- ✅ **{stage}**: Completed in {duration:.2f}s\n"
                elif status == "failed":
                    content += f"- ❌ **{stage}**: Failed\n"
                else:
                    content += f"- ⏳ **{stage}**: Not started\n"

            content += f"""
## 🔧 Device Information
- **Device Used**: {summary['device_information']['recommended_device'].upper()}
- **CUDA Available**: {'Yes' if summary['device_information']['cuda_available'] else 'No'}
- **GPU Count**: {summary['device_information']['gpu_count']}

## 📋 Recommendations
"""

            for rec in summary["recommendations"]:
                content += f"- {rec}\n"

            content += f"""
## 🚀 Next Steps
"""

            for step in summary["next_steps"]:
                content += f"- {step}\n"

            content += f"""
---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"📝 Human-readable summary saved: {summary_path}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create human-readable summary: {e}")


def main():
    """Main entry point for the workflow runner."""
    parser = argparse.ArgumentParser(
        description="LawBot Training Pipeline Workflow Runner"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES.keys()),
        help="Specific stages to run",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a predefined stage preset",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart all stages, ignoring checkpoints",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all available stages and presets",
    )

    args = parser.parse_args()

    # SỬA: Setup workflow logging with consistent timestamp
    workflow_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging("workflow", workflow_timestamp)
    logger = get_logger("main")

    # SỬA: Log system information and cleanup old logs and reports
    from core.utils.logging_manager import cleanup_logs, log_system_info
    from core.utils.report_manager import cleanup_old_reports, get_report_summary
    
    log_system_info()
    cleanup_logs(max_workflow_logs=3)  # Keep only 3 latest workflow logs
    
    # SỬA: Cleanup old reports
    cleanup_old_reports(keep_count=2, max_age_days=3)  # Keep 2 latest per type, max 3 days old
    
    # Log report summary
    report_summary = get_report_summary()
    logger.info(f"📊 Report summary: {report_summary['total_reports']} files, {report_summary['total_size_mb']:.2f} MB")

    # List stages and presets if requested
    if args.list_stages:
        print("Available Stages:")
        for stage, info in STAGES.items():
            print(f"  {stage}: {info['description']}")
            if info["dependencies"]:
                print(f"    Dependencies: {', '.join(info['dependencies'])}")

        print("\nAvailable Presets:")
        for preset, stages in PRESETS.items():
            print(f"  {preset}: {', '.join(stages)}")
        return 0

    # Determine stages to run
    if args.preset:
        if args.preset not in PRESETS:
            logger.error(f"Unknown preset: {args.preset}")
            return 1
        stages = PRESETS[args.preset]
    elif args.stages:
        stages = args.stages
    else:
        # Default to full preset
        stages = PRESETS["full"]

    # Create and run workflow
    try:
        # Log 3-tier architecture information
        logger.info("🏗️ LawBot 3-Tier Architecture Pipeline:")
        logger.info(
            "  Tier 1 (Retrieval): Bi-Encoder with Contrastive Learning + ADAPT + HPO + HNM"
        )
        logger.info(
            "  Tier 2 (Light Reranking): PhoBERT-base-v2 with independent ADAPT training"
        )
        logger.info(
            "  Tier 3 (Cross-Encoder): Dual ADAPT Ensemble (PhoBERT-base-v2 + PhoBERT-large)"
        )
        logger.info(
            "  Model Inheritance: Tier 1 → Tier 2 → Tier 3 (inherits from Tier 2)"
        )

        # CRITICAL: Log data source information
        logger.info("📊 Data Source Information:")
        try:
            from config.paths import validate_training_data_paths, TRAINING_DATA_PATHS

            # Get validation results
            validation_results = validate_training_data_paths()

            # Log primary paths
            primary_paths = TRAINING_DATA_PATHS["primary"]
            logger.info("  🔍 Primary training data paths:")
            for data_type, path in primary_paths.items():
                status = "✅ EXISTS" if path.exists() else "❌ MISSING"
                logger.info(f"    {data_type}: {path} - {status}")

            # Log validation status
            logger.info("  📊 Data validation status:")
            for tier_name, tier_info in validation_results.items():
                if tier_name != "overall":
                    status = tier_info["status"]
                    data_source = tier_info["data_source"]
                    file_count = tier_info["file_count"]
                    logger.info(
                        f"    {tier_name}: {status} ({data_source}, {file_count} files)"
                    )

            # Overall status
            overall_status = validation_results["overall"]
            if overall_status["real_data_available"]:
                logger.info("  ✅ All tiers have real training data available")
                logger.info(
                    "  💡 Real data from run_preparation.py will be used for optimal performance"
                )
            else:
                logger.warning("  ⚠️ Some tiers are missing real training data")
                logger.warning(
                    "  💡 This will result in poor performance - run data_preparation.py first"
                )

        except ImportError as e:
            logger.warning(f"⚠️ Could not import centralized paths config: {e}")
            logger.info(
                "  🔍 Training data will be loaded from: features/processed_data/"
            )
            logger.info(
                "  📁 Expected files: bi_encoder_train.jsonl, cross_encoder_train.jsonl"
            )
            logger.info(
                "  💡 Real data from run_preparation.py is REQUIRED for good performance"
            )
            logger.info(
                "  ⚠️  No synthetic fallback is available; pipeline will abort if real data is missing"
            )
        except Exception as e:
            logger.warning(f"⚠️ Centralized paths validation failed: {e}")
            logger.info(
                "  🔍 Training data will be loaded from: features/processed_data/"
            )
            logger.info(
                "  📁 Expected files: bi_encoder_train.jsonl, cross_encoder_train.jsonl"
            )
            logger.info(
                "  💡 Real data from run_preparation.py is REQUIRED for good performance"
            )
            logger.info(
                "  ⚠️  No synthetic fallback is available; pipeline will abort if real data is missing"
            )

        # Log execution consistency information
        logger.info(
            "🔍 Execution Consistency: Direct import method (identical to manual execution)"
        )
        logger.info(
            "🔧 Environment: No modifications (preserves manual execution context)"
        )
        logger.info(
            "📂 Working Directory: Current directory (same as manual execution)"
        )
        logger.info("⚡ Performance: Direct execution (no subprocess overhead)")

        runner = WorkflowRunner(stages, force_restart=args.force_restart)
        success = runner.run_workflow()
        return 0 if success else 1

    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
