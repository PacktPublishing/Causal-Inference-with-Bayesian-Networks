#!/bin/bash

# Deactivate any active virtual environment
deactivate 2>/dev/null || true

# Activate the virtual environment in the correct project folder
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/.venv/bin/activate"

# Print the Python executable to verify it's using the correct virtual environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Run a simple test to verify the environment is working correctly
python -c "import sys; print('Python path:', sys.executable)"
