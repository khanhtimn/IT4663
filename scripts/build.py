#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)


def create_virtual_env():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    else:
        print("Virtual environment already exists")


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        sys.exit(1)

    # Determine the pip executable path
    if os.name == "nt":  # Windows
        pip_path = Path("venv/Scripts/pip")
    else:  # Unix-like
        pip_path = Path("venv/bin/pip")

    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)


def main():
    """Main build function."""
    print("Starting build process...")

    # Check Python version
    check_python_version()

    # Create virtual environment
    create_virtual_env()

    # Install dependencies
    install_dependencies()

    print("\nBuild completed successfully!")
    print("\nTo activate the virtual environment:")
    if os.name == "nt":  # Windows
        print("    .\\venv\\Scripts\\activate")
    else:  # Unix-like
        print("    source venv/bin/activate")


if __name__ == "__main__":
    main()
