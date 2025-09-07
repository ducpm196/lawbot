#!/usr/bin/env python3
"""
Hyperparameter Optimization - LawBot v8.2
========================================

Hyperparameter optimization for LawBot models using Optuna.
"""

import sys
from pathlib import Path
import logging
import argparse
from typing import Dict, Any, Optional
import pickle
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import optuna
from torch.optim import Adam

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from training.engine import TrainingEngine
    from core.utils.io import load_jsonl
    from core.datasets import RerankerDataset
    from core.utils.logging_manager import setup_logging
    from config.loader import config
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the hyperparameter optimizer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.study = None

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default HPO configuration."""
        # Default HPO configuration
        hpo_config = {
            "n_trials": 20,
            "learning_rate_range": [1e-6, 1e-4],
            "batch_size_options": [8, 16, 32],
            "max_epochs_per_trial": 1,
            "optimization_direction": "maximize",
            "metric": "validation_accuracy",
        }

        # Note: config.hpo is not defined in the Pydantic schema
        # Using default configuration only

        return hpo_config

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation accuracy score
        """
        # Suggest hyperparameters
        learning_rate = trial.suggest_float(
            "learning_rate",
            self._get_default_config()["learning_rate_range"][0],
            self._get_default_config()["learning_rate_range"][1],
            log=True,
        )
        batch_size = trial.suggest_categorical(
            "batch_size", self._get_default_config()["batch_size_options"]
        )
        num_epochs = trial.suggest_int(
            "num_epochs", 1, self._get_default_config()["max_epochs_per_trial"]
        )

        try:
            # Load training data
            train_data = self._load_training_data()
            if not train_data:
                logger.error("No training data available")
                return 0.0

            # Create validation split
            val_data = self._create_validation_split(train_data, split_ratio=0.2)

            # Initialize training engine
            engine = TrainingEngine(
                model_name="vinai/phobert-base-v2",
                train_dataset=train_data,
                eval_dataset=val_data,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                train_batch_size=batch_size,
            )

            # Setup model and training components
            engine.setup_model_and_tokenizer()
            engine.setup_optimizer_and_scheduler(len(train_data) // batch_size)

            # Train model
            training_results = engine.train()

            # Return validation accuracy
            return training_results.get("eval_accuracy", 0.0)

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0

    def _load_training_data(self) -> list:
        """Load training data."""
        try:
            # Try to load from default location
            data_path = Path("features/processed_data")
            if data_path.exists():
                train_file = data_path / "cross_encoder_train.jsonl"
                if train_file.exists():
                    return load_jsonl(train_file)

            logger.warning("No training data found, using empty list")
            return []

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return []

    def _create_validation_split(self, data: list, split_ratio: float = 0.2) -> list:
        """Create validation split from training data."""
        split_idx = int(len(data) * (1 - split_ratio))
        return data[split_idx:]

    def _get_reranker_config(self) -> Dict[str, Any]:
        """Get reranker configuration."""
        try:
            return config.reranker_pipeline.combined_reranker.model_dump()
        except:
            return {}

    def optimize(self, n_trials: int = None) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run

        Returns:
            Optimization results
        """
        if n_trials is None:
            n_trials = self._get_default_config()["n_trials"]

        logger.info(f"Starting HPO with {n_trials} trials...")

        # Create study
        study_name = f"lawbot_hpo_{Path.cwd().name}"
        self.study = optuna.create_study(
            direction=self._get_default_config()["optimization_direction"],
            study_name=study_name,
        )

        # Run optimization
        self.study.optimize(self.objective, n_trials=n_trials)

        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value

        logger.info(f"Best trial: {best_value}")
        logger.info(f"Best parameters: {best_params}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": self.study,
        }

    def save_results(self, output_path: str):
        """Save optimization results.

        Args:
            output_path: Path to save results
        """
        if not self.study:
            logger.warning("No study to save")
            return

        try:
            # Save study
            study_path = Path(output_path) / "hpo_study.pkl"
            with open(study_path, "wb") as f:
                pickle.dump(self.study, f)

            # Save best parameters
            best_params_path = Path(output_path) / "best_params.json"
            with open(best_params_path, "w") as f:
                json.dump(self.study.best_params, f, indent=2)

            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point for HPO."""
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    parser.add_argument(
        "--output", type=str, default="hpo_results", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Run optimization
    optimizer = HyperparameterOptimizer()
    results = optimizer.optimize(n_trials=args.n_trials)

    # Save results
    optimizer.save_results(str(output_dir))

    logger.info("Hyperparameter optimization completed!")


if __name__ == "__main__":
    main()
