#!/usr/bin/env python3
"""
LawBot Application Launcher
==========================

Launches the LawBot Streamlit application with proper logging configuration.
This script ensures the app runs from the correct working directory and sets up logging.
"""

import subprocess
import sys
from pathlib import Path
import os


def run_app(port: int = 8501, config_file: str = None):
    """Launch the LawBot Streamlit application."""
    # Get project root directory
    project_root = Path(__file__).resolve().parent
    app_file = project_root / "app" / "app.py"

    # Check if app file exists
    if not app_file.exists():
        print(f"❌ Lỗi: Không tìm thấy tệp ứng dụng tại '{app_file}'.")
        return 1

    # Setup app logging only when launching the app
    try:
        from core.utils.logging_manager import setup_logging

        setup_logging("app")
        print("✅ App logging configured successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not setup app logging: {e}")

    print(f"🚀 Đang khởi chạy ứng dụng LawBot từ: {app_file}")
    print(f"📁 Working directory: {project_root}")

    # Build command
    cmd = [
        str(project_root / "venv" / "Scripts" / "python.exe"),
        "-m",
        "streamlit",
        "run",
        str(app_file),
        "--server.port",
        str(port),
    ]

    if config_file:
        cmd.extend(["--config", config_file])

    print(f"🚀 Command: {' '.join(cmd)}")

    # Set environment variables for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["LAWBOT_PROJECT_ROOT"] = str(project_root)

    try:
        # Run the app
        result = subprocess.run(cmd, cwd=project_root, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy ứng dụng: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 Ứng dụng đã được dừng bởi người dùng")
        return 0
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch LawBot Streamlit Application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on")
    parser.add_argument("--config", help="Path to Streamlit config file")

    args = parser.parse_args()

    return run_app(port=args.port, config_file=args.config)


if __name__ == "__main__":
    sys.exit(main())
