#!/usr/bin/env python3
"""
Clear Notebook Outputs Script
=============================

This script clears the outputs from Jupyter notebooks to remove any personal paths
or sensitive information before public release.
"""

import os
import json
import glob

def clear_notebook_outputs(notebook_path):
    """Clear the outputs from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check if it's a valid notebook
    if 'cells' not in notebook:
        print(f"Warning: {notebook_path} does not appear to be a valid notebook")
        return
    
    # Clear outputs from all cells
    for cell in notebook['cells']:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    
    # Write the modified notebook back to disk
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Cleared outputs from {notebook_path}")

def main():
    """Clear outputs from all notebooks in the notebooks directory."""
    notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))
    
    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        return
    
    print(f"Found {len(notebooks)} notebooks")
    for notebook in notebooks:
        clear_notebook_outputs(notebook)
    
    print("All notebook outputs cleared")

if __name__ == "__main__":
    main()