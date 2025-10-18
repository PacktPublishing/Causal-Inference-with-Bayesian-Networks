# Project: Chapter 04 / SEM Demo

This project is prepared to use the semopy library for Structural Equation Modeling (SEM).

## Setup

- Ensure you have Python 3.9+ (3.10–3.13 recommended).
- Create and activate a virtual environment.
  - macOS/Linux:
    python -m venv .venv
    source .venv/bin/activate
  - Windows (PowerShell):
    python -m venv .venv
    .venv\Scripts\Activate.ps1
- Install dependencies:
  pip install -r requirements.txt

## Run the demo

- Execute the minimal example to verify the environment:
  python use_semopy_demo.py
- It will fit a simple one-factor CFA model on simulated data and print parameter estimates and selected fit indices.
- If optional report dependencies are available, it will generate an HTML report named sem_report.html in the project directory.

## Notes

- The .idea directory is included for JetBrains IDE settings. To ensure a valid interpreter:
  - In PyCharm: Settings/Preferences → Project → Python Interpreter → Add → Existing environment → select .venv/bin/python (macOS/Linux) or .venv\\Scripts\\python.exe (Windows).
  - Alternatively, create a new virtual environment from the IDE and install requirements.txt.
- semopy will install required dependencies; if installation fails, make sure system build tools and a recent version of pip are present:
  python -m pip install --upgrade pip setuptools wheel


## Documentation

- See MODEL_SPEC_AND_STEPS.md for the detailed Political Democracy model specification, estimation steps with semopy, and the explanation of why the path diagram shows six residual-correlation arcs while the model specification includes only four.
- See SEMOPY_TUTORIAL_WALKTHROUGH.md for a step-by-step code and theory walkthrough of `semopy_demo_tutorial.ipynb`.

## Tutorial Notebook

- For a guided, step-by-step introduction, open and run `semopy_demo_tutorial.ipynb`. It simulates a small CFA dataset, fits the model with semopy, explains each step in Markdown cells, and draws a path diagram in a style similar to the `bollen_semopy.ipynb` workbook.
