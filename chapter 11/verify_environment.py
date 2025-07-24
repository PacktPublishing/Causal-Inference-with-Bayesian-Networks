#!/usr/bin/env python3
"""
Verify Environment Script
========================

This script verifies that the Python environment is correctly set up for the
Causal Inference with Machine Learning project. It checks:

1. The Python executable path to ensure it's using the correct virtual environment
2. The ability to import required modules
3. The correct working directory

Run this script after activating the virtual environment to verify everything is set up correctly.
"""

import sys
import os
import importlib
import platform

def check_python_executable():
    """Check if the Python executable is in the correct location."""
    executable = sys.executable
    expected_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")

    print(f"Python executable: {executable}")

    if expected_prefix in executable:
        print("✅ Using the correct virtual environment")
        return True
    else:
        print("❌ Not using the expected virtual environment")
        print(f"Expected path should contain: {expected_prefix}")
        return False

def check_imports():
    """Check if required modules can be imported."""
    required_modules = [
        "numpy", 
        "pandas", 
        "matplotlib", 
        "seaborn", 
        "sklearn",
        "metalearners.s_learner",
        "metalearners.t_learner",
        "metalearners.x_learner",
        "metalearners.r_learner",
        "synthetic_data.synthetic_data_for_cate2"
    ]

    all_imports_successful = True
    print("\nChecking imports:")

    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ Successfully imported {module}")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            all_imports_successful = False

    return all_imports_successful

def check_working_directory():
    """Check if the working directory is correct."""
    current_dir = os.getcwd()
    expected_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\nCurrent working directory: {current_dir}")

    if current_dir == expected_dir:
        print("✅ Working in the correct directory")
        return True
    else:
        print("❌ Not in the expected directory")
        print(f"Expected directory: {expected_dir}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 80)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 80)

    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    executable_check = check_python_executable()
    imports_check = check_imports()
    directory_check = check_working_directory()

    print("\n" + "=" * 80)
    if executable_check and imports_check and directory_check:
        print("✅ All checks passed! Your environment is correctly set up.")
    else:
        print("❌ Some checks failed. Please review the issues above.")
    print("=" * 80)

if __name__ == "__main__":
    main()
