#!/usr/bin/env python3
"""
Training Evaluator - LawBot v8.1
================================

Automated evaluation system for monitoring training progress and model performance
during the training process.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from core.utils.logging_manager import get_logger
from core.utils.io import ensure_directory_exists

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for each epoch/step."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: str = ""


@dataclass
class ModelEvaluation:
    """Model evaluation results."""
    model_path: str
    model_version: str
    evaluation_metrics: Dict[str, float]
    training_metrics: List[TrainingMetrics]
    validation_results: Dict[str, Any]
    timestamp: str
    training_duration: float
    best_epoch: int
    convergence_status: str


class TrainingEvaluator:
    """Automated training evaluation system."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize training evaluator."""
        self.output_dir = output_dir or Path("training/evaluation_logs")
        ensure_directory_exists(self.output_dir)
        
        self.training_history: List[TrainingMetrics] = []
        self.model_evaluations: List[ModelEvaluation] = []
        self.current_training_session: Optional[str] = None
        
        logger.info("✅ Training Evaluator initialized")
    
    def start_training_session(self, model_name: str, model_type: str) -> str:
        """Start a new training session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{model_name}_{model_type}_{timestamp}"
        self.current_training_session = session_id
        
        # Create session directory
        session_dir = self.output_dir / session_id
        ensure_directory_exists(session_dir)
        
        logger.info(f"🚀 Started training session: {session_id}")
        return session_id
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        accuracy: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        memory_usage_mb: float = 0.0
    ):
        """Log training step metrics."""
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            memory_usage_mb=memory_usage_mb,
            timestamp=datetime.now().isoformat()
        )
        
        self.training_history.append(metrics)
        
        # Log every 100 steps to avoid spam
        if step % 100 == 0:
            logger.info(
                f"📊 Epoch {epoch}, Step {step}: Loss={loss:.4f}, "
                f"LR={learning_rate:.6f}, Acc={accuracy:.3f}, "
                f"F1={f1_score:.3f}"
            )
    
    def log_validation_step(
        self,
        epoch: int,
        validation_loss: float,
        validation_accuracy: float,
        validation_metrics: Optional[Dict[str, float]] = None
    ):
        """Log validation step metrics."""
        # Find the last training step for this epoch
        for metrics in reversed(self.training_history):
            if metrics.epoch == epoch:
                metrics.validation_loss = validation_loss
                metrics.validation_accuracy = validation_accuracy
                break
        
        logger.info(
            f"🔍 Validation Epoch {epoch}: Loss={validation_loss:.4f}, "
            f"Acc={validation_accuracy:.3f}"
        )
    
    def evaluate_model_performance(
        self,
        model_path: str,
        model_version: str,
        validation_data: List[Dict[str, Any]],
        training_duration: float,
        best_epoch: int
    ) -> ModelEvaluation:
        """Evaluate model performance on validation data."""
        logger.info(f"🔍 Evaluating model: {model_path}")
        
        # Calculate overall training metrics
        if self.training_history:
            avg_loss = np.mean([m.loss for m in self.training_history])
            avg_accuracy = np.mean([m.accuracy for m in self.training_history])
            avg_f1 = np.mean([m.f1_score for m in self.training_history])
            
            # Check convergence
            recent_losses = [m.loss for m in self.training_history[-50:]]
            if len(recent_losses) >= 10:
                loss_std = np.std(recent_losses)
                convergence_status = "converged" if loss_std < 0.01 else "converging"
            else:
                convergence_status = "early_training"
        else:
            avg_loss = avg_accuracy = avg_f1 = 0.0
            convergence_status = "no_data"
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(validation_data)
        
        # Create evaluation result
        evaluation = ModelEvaluation(
            model_path=model_path,
            model_version=model_version,
            evaluation_metrics={
                "avg_training_loss": avg_loss,
                "avg_training_accuracy": avg_accuracy,
                "avg_training_f1": avg_f1,
                **validation_metrics
            },
            training_metrics=self.training_history.copy(),
            validation_results={"data": validation_data},
            timestamp=datetime.now().isoformat(),
            training_duration=training_duration,
            best_epoch=best_epoch,
            convergence_status=convergence_status
        )
        
        self.model_evaluations.append(evaluation)
        
        # Save evaluation
        self._save_evaluation(evaluation)
        
        logger.info(f"✅ Model evaluation completed: {convergence_status}")
        return evaluation
    
    def _calculate_validation_metrics(self, validation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate validation metrics."""
        if not validation_data:
            return {"validation_accuracy": 0.0, "validation_f1": 0.0}
        
        # Simple accuracy calculation for now
        # Can be enhanced with more sophisticated metrics
        correct_predictions = 0
        total_predictions = len(validation_data)
        
        for item in validation_data:
            if item.get("correct", False):
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "validation_accuracy": accuracy,
            "validation_f1": accuracy  # Simplified for now
        }
    
    def _save_evaluation(self, evaluation: ModelEvaluation):
        """Save evaluation results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_evaluation_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert dataclass to dict
            eval_dict = asdict(evaluation)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(eval_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Evaluation saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save evaluation: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary for monitoring."""
        if not self.training_history:
            return {"status": "no_training_data"}
        
        # Calculate training statistics
        losses = [m.loss for m in self.training_history]
        accuracies = [m.accuracy for m in self.training_history]
        f1_scores = [m.f1_score for m in self.training_history]
        
        summary = {
            "status": "active",
            "total_steps": len(self.training_history),
            "total_epochs": max(m.epoch for m in self.training_history) if self.training_history else 0,
            "current_loss": losses[-1] if losses else 0.0,
            "best_loss": min(losses) if losses else 0.0,
            "current_accuracy": accuracies[-1] if accuracies else 0.0,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
            "current_f1": f1_scores[-1] if f1_scores else 0.0,
            "best_f1": max(f1_scores) if f1_scores else 0.0,
            "loss_trend": "decreasing" if len(losses) > 1 and losses[-1] < losses[-2] else "increasing",
            "convergence_status": self._assess_convergence(),
            "last_updated": self.training_history[-1].timestamp if self.training_history else None
        }
        
        return summary
    
    def _assess_convergence(self) -> str:
        """Assess training convergence status."""
        if len(self.training_history) < 10:
            return "early_training"
        
        recent_losses = [m.loss for m in self.training_history[-20:]]
        if len(recent_losses) < 10:
            return "insufficient_data"
        
        # Calculate loss stability
        loss_std = np.std(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        if loss_std < 0.01 and abs(loss_trend) < 0.001:
            return "converged"
        elif loss_std < 0.05:
            return "converging"
        else:
            return "not_converging"
    
    def export_training_report(self, output_path: Optional[Path] = None) -> bool:
        """SỬA: Export comprehensive training report only if there's actual training data."""
        try:
            # SỬA: Check if there's actual training data
            if not self.training_history and not self.model_evaluations:
                logger.warning("⚠️ No training data to export, skipping training report")
                return False
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.output_dir / f"training_report_{timestamp}.json"
            
            ensure_directory_exists(output_path.parent)
            
            report = {
                "training_summary": self.get_training_summary(),
                "training_history": [asdict(m) for m in self.training_history],
                "model_evaluations": [asdict(e) for e in self.model_evaluations],
                "current_session": self.current_training_session,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Training report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export training report: {e}")
            return False
    
    def plot_training_curves(self) -> Dict[str, Any]:
        """Generate training curves data for visualization."""
        if not self.training_history:
            return {"status": "no_data"}
        
        # Prepare data for plotting
        epochs = [m.epoch for m in self.training_history]
        losses = [m.loss for m in self.training_history]
        accuracies = [m.accuracy for m in self.training_history]
        f1_scores = [m.f1_score for m in self.training_history]
        
        # Validation data if available
        val_losses = []
        val_accuracies = []
        for m in self.training_history:
            val_losses.append(m.validation_loss if m.validation_loss is not None else None)
            val_accuracies.append(m.validation_accuracy if m.validation_accuracy is not None else None)
        
        return {
            "status": "success",
            "epochs": epochs,
            "training_loss": losses,
            "training_accuracy": accuracies,
            "training_f1": f1_scores,
            "validation_loss": val_losses,
            "validation_accuracy": val_accuracies,
            "convergence_status": self._assess_convergence()
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Export final report if training session exists
            if self.current_training_session:
                self.export_training_report()
            
            logger.info("🧹 Training Evaluator cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")


# Global instance for easy access
training_evaluator = TrainingEvaluator()
