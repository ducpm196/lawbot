import argparse
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import sys

from core.utils.logging_manager import setup_logging

"""
Base Training Script - LawBot v8.0
==================================

Base class for all training scripts providing common functionality
and argument parsing.
"""

logger = logging.getLogger(__name__)


class BaseTrainingScript:
    """Base class for all training scripts."""

    def __init__(self, script_name: str):
        """Initialize the base training script.

        Args:
            script_name: Name of the training script
        """
        self.script_name = script_name
        self.num_training_examples = 0
        self.data_path: Optional[Path] = None

    def run(self):
        """Main entry point for training scripts."""
        try:
            # Parse command line arguments
            parser = self._create_argument_parser()
            args = parser.parse_args()

            # Setup logging
            setup_logging()
            logger.info(f"Starting {self.script_name} training script")

            # Auto-find data path if not provided
            if args.data_path is None:
                try:
                    args.data_path = self.find_latest_data_path()
                except FileNotFoundError as e:
                    logger.error(f"Could not find data path: {e}")
                    sys.exit(1)

            logger.info(f"Data path: {args.data_path}")

            # Validate data path
            if not args.data_path.exists():
                raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

            if not args.data_path.is_dir():
                raise NotADirectoryError(
                    f"Data path is not a directory: {args.data_path}"
                )

            # Store data path for later use
            self.data_path = args.data_path

            # Run training
            engine = self.train(args.data_path)

            # Save model
            if args.output_dir:
                output_dir = args.output_dir
            else:
                output_dir = None  # Will use default timestamped path

            saved_path = self.save_model(engine, output_dir)

            # Get and log metadata
            metadata = self.get_metadata(args.data_path, engine)
            logger.info("Training completed successfully")
            logger.info(f"Model saved to: {saved_path}")
            logger.info(f"Metadata: {metadata}")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for the training script.

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description=f"Training script for {self.script_name}"
        )
        parser.add_argument(
            "--data-path",
            type=Path,
            help="Path to processed data directory (auto-detected if not provided)",
        )
        parser.add_argument(
            "--output-dir", type=Path, help="Output directory for model"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose logging"
        )

        return parser

    def find_latest_data_path(self) -> Path:
        """Find the latest processed data directory.

        Returns:
            Path to the latest data directory

        Raises:
            FileNotFoundError: If no data directory is found
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement find_latest_data_path")

    def train(self, data_path: Path):
        """Run the training process.

        Args:
            data_path: Path to the training data

        Returns:
            Trained model engine

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement train")

    def save_model(self, engine, output_dir: Optional[Path] = None) -> Path:
        """Save the trained model.

        Args:
            engine: Trained model engine
            output_dir: Output directory (optional)

        Returns:
            Path where the model was saved

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement save_model")

    def get_metadata(self, data_path: Path, engine) -> Dict[str, Any]:
        """Get metadata about the training run.

        Args:
            data_path: Path to the training data
            engine: Trained model engine

        Returns:
            Dictionary containing training metadata

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_metadata")

    def validate_data(self, data_path: Path) -> bool:
        """Validate the training data.

        Args:
            data_path: Path to the training data

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Basic validation - check if directory exists and contains files
            if not data_path.exists():
                logger.error(f"Data path does not exist: {data_path}")
                return False

            if not data_path.is_dir():
                logger.error(f"Data path is not a directory: {data_path}")
                return False

            # Check if directory contains any files
            files = list(data_path.glob("*"))
            if not files:
                logger.error(f"Data directory is empty: {data_path}")
                return False

            logger.info(f"Data validation passed. Found {len(files)} files/directories")
            return True

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
