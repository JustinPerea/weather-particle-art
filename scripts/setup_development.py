#!/usr/bin/env python3
"""Set up development environment for weather particle art."""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Ensure Python 3.9+ is available."""
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def create_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("venv")

    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])

    # Activation instructions
    if platform.system() == "Windows":
        activate = "venv\\Scripts\\activate"
    else:
        activate = "source venv/bin/activate"

    print(f"\nTo activate virtual environment:")
    print(f"  {activate}")

def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_gpu():
    """Check for CUDA-capable GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠ No CUDA GPU detected - performance will be limited")
    except ImportError:
        print("⚠ PyTorch not installed - GPU detection skipped")

def main():
    """Run complete development setup."""
    print("=== Weather Particle Art Development Setup ===\n")

    check_python_version()
    create_virtual_environment()
    # Note: User must activate venv before installing dependencies

    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Activate virtual environment")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run verification: python scripts/run_verification.py")
    print("\nHappy developing! 🎨")

if __name__ == "__main__":
    main()
