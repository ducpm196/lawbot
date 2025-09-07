#!/usr/bin/env python3
"""
LawBot Project Setup Script
===========================

Comprehensive setup script that automates the entire project setup process
from environment setup to running the application.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    """Automated project setup for LawBot."""

    def __init__(self, project_root: Path):
        """Initialize the project setup."""
        self.project_root = project_root
        self.venv_path = project_root / "venv"
        self.requirements_file = project_root / "requirements.txt"

    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            logger.info(
                f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible"
            )
            return True
        else:
            logger.error(
                f"❌ Python {version.major}.{version.minor} is too old. Need 3.8+"
            )
            return False

    def create_virtual_environment(self) -> bool:
        """Create virtual environment."""
        try:
            if self.venv_path.exists():
                logger.info("Virtual environment already exists")
                return True

            logger.info("Creating virtual environment...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)], check=True
            )
            logger.info("✅ Virtual environment created successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False

    def get_activate_command(self) -> str:
        """Get the appropriate activate command for the OS."""
        if os.name == "nt":  # Windows
            return str(self.venv_path / "Scripts" / "activate")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "activate")

    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        try:
            if not self.requirements_file.exists():
                logger.error("❌ requirements.txt not found")
                return False

            logger.info("Installing dependencies...")

            # Use pip from virtual environment
            if os.name == "nt":  # Windows
                pip_path = self.venv_path / "Scripts" / "pip"
            else:  # Unix/Linux/macOS
                pip_path = self.venv_path / "bin" / "pip"

            subprocess.run(
                [str(pip_path), "install", "-r", str(self.requirements_file)],
                check=True,
            )
            logger.info("✅ Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False

    def create_directory_structure(self) -> bool:
        """Create necessary directory structure."""
        try:
            directories = [
                "data/raw",
                "models",
                "features",
                "reports",
                "logs",
                "checkpoints",
                "training",
                "evaluation",
            ]

            for dir_path in directories:
                full_path = self.project_root / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False

    def run_environment_check(self) -> bool:
        """Run environment health check."""
        try:
            logger.info("Running environment check...")
            result = subprocess.run(
                [sys.executable, "scripts/check_environment.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ Environment check passed")
                return True
            else:
                logger.warning(f"⚠️ Environment check warnings: {result.stdout}")
                return True  # Continue even with warnings

        except Exception as e:
            logger.error(f"❌ Environment check failed: {e}")
            return False

    def run_project_check(self) -> bool:
        """Run project structure check."""
        try:
            logger.info("Running project structure check...")
            result = subprocess.run(
                [sys.executable, "scripts/utils/check_project.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ Project structure check passed")
                return True
            else:
                logger.warning(f"⚠️ Project structure check warnings: {result.stdout}")
                return True  # Continue even with warnings

        except Exception as e:
            logger.error(f"❌ Project structure check failed: {e}")
            return False

    def check_aid_alignment(self) -> bool:
        """Check AID alignment."""
        try:
            logger.info("Checking AID alignment...")
            result = subprocess.run(
                [sys.executable, "scripts/utils/check_aid_alignment.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ AID alignment check passed")
                return True
            else:
                logger.warning(f"⚠️ AID alignment check warnings: {result.stdout}")
                return True  # Continue even with warnings

        except Exception as e:
            logger.error(f"❌ AID alignment check failed: {e}")
            return False

    def run_data_preparation(self) -> bool:
        """Run data preparation if data exists."""
        try:
            data_dir = self.project_root / "data" / "raw"
            if not any(data_dir.iterdir()):
                logger.info("No data files found, skipping data preparation")
                return True

            logger.info("Running data preparation...")
            result = subprocess.run(
                [sys.executable, "data_processing/run_preparation.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ Data preparation completed")
                return True
            else:
                logger.warning(f"⚠️ Data preparation warnings: {result.stdout}")
                return True  # Continue even with warnings

        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False

    def run_workflow(self, preset: str = "full") -> bool:
        """Run the training workflow."""
        try:
            logger.info(f"Running workflow with preset: {preset}")
            result = subprocess.run(
                [sys.executable, "run_workflow.py", "--preset", preset],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ Workflow completed successfully")
                return True
            else:
                logger.error(f"❌ Workflow failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return False

    def run_app(self) -> bool:
        """Run the Streamlit application."""
        try:
            logger.info("Starting Streamlit application...")

            # Use the app runner script
            result = subprocess.run(
                [sys.executable, "run_app.py"], cwd=self.project_root
            )

            return True

        except KeyboardInterrupt:
            logger.info("App stopped by user")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to run app: {e}")
            return False

    def setup_project(
        self, skip_training: bool = False, workflow_preset: str = "full"
    ) -> bool:
        """Complete project setup."""
        logger.info("🚀 Starting LawBot project setup...")

        # Check Python version
        if not self.check_python_version():
            return False

        # Create virtual environment
        if not self.create_virtual_environment():
            return False

        # Install dependencies
        if not self.install_dependencies():
            return False

        # Create directory structure
        if not self.create_directory_structure():
            return False

        # Run checks
        if not self.run_environment_check():
            return False

        if not self.run_project_check():
            return False

        if not self.check_aid_alignment():
            return False

        # Data preparation
        if not self.run_data_preparation():
            return False

        # Training workflow (optional)
        if not skip_training:
            if not self.run_workflow(workflow_preset):
                logger.warning("⚠️ Training workflow failed, but setup can continue")

        logger.info("✅ Project setup completed successfully!")
        return True

    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📋 Next steps:")
        print("1. Activate virtual environment:")
        if os.name == "nt":  # Windows
            print(f"   {self.venv_path}\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print(f"   source {self.venv_path}/bin/activate")

        print("\n2. Run the application:")
        print("   python run_app.py")

        print("\n3. Or run training workflow:")
        print("   python run_workflow.py --preset full")

        print("\n4. Access the app at: http://localhost:8501")
        print("\n📚 For more information, see README.md")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LawBot Project Setup")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training workflow during setup",
    )
    parser.add_argument(
        "--workflow-preset",
        default="full",
        choices=["full", "training", "retrieval", "reranking", "evaluation_only"],
        help="Workflow preset to run (default: full)",
    )
    parser.add_argument(
        "--run-app", action="store_true", help="Run the app after setup"
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent

    # Create setup instance
    setup = ProjectSetup(project_root)

    # Run setup
    if setup.setup_project(args.skip_training, args.workflow_preset):
        setup.print_next_steps()

        # Run app if requested
        if args.run_app:
            logger.info("Starting application...")
            setup.run_app()
    else:
        logger.error("❌ Project setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
