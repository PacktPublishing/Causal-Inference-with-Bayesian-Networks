"""
Minimal demo showing how to use the semopy library.

Usage:
  python use_semopy_demo.py

It builds a small measurement model, simulates data, fits the model, and prints a summary.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

try:
    import semopy
    from semopy import Model
    try:
        from semopy import report as sem_report  # optional
    except Exception:
        sem_report = None
except Exception as e:  # pragma: no cover - import guidance for users
    raise SystemExit(
        "semopy is not installed. Please run:\n"
        "  pip install -r requirements.txt\n\n"
        f"Original error: {e}"
    )


def build_sample_data(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Latent variable eta with variance 1
    eta = rng.normal(size=n)
    # Loadings for three indicators y1, y2, y3
    y1 = 0.8 * eta + rng.normal(scale=0.6, size=n)
    y2 = 0.9 * eta + rng.normal(scale=0.5, size=n)
    y3 = 0.7 * eta + rng.normal(scale=0.7, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})
    return df


# Simple one-factor confirmatory factor analysis (CFA) model in lavaan-like syntax
MODEL_SPEC = """
# Measurement model
eta =~ y1 + y2 + y3

# Fix latent variance to 1 for identifiability (optional; semopy can handle scaling automatically)
eta ~~ 1*eta
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal semopy CFA demo")
    parser.add_argument("--n", type=int, default=120, help="Number of simulated observations")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--report", action="store_true", help="Generate HTML report (may require extra deps)")
    args = parser.parse_args()

    data = build_sample_data(n=args.n, seed=args.seed)
    model = Model(MODEL_SPEC)

    # Fit using default optimizer; keep it simple to avoid heavy resource use
    try:
        model.fit(data)
    except Exception as e:
        # Some environments may need a different stats function; try a basic fallback
        raise SystemExit(f"Model fitting failed: {e}")

    # Print parameter estimates (compatibility across semopy versions)
    try:
        estimates = model.inspect()  # preferred
    except Exception:
        if hasattr(semopy, "inspect"):
            estimates = semopy.inspect(model)
        else:
            estimates = getattr(model, "param_vals", None)
    print("Parameter estimates:\n", estimates)

    # Compute and print some fit indices (compatibility across semopy versions)
    stats = None
    try:
        if hasattr(model, "calc_stats"):
            stats = model.calc_stats()
        else:
            stats = semopy.calc_stats(model)
    except TypeError:
        try:
            stats = semopy.calc_stats(model, data)
        except Exception:
            stats = None
    except Exception:
        stats = None

    if stats is None:
        print("\nFit statistics: unavailable for this semopy version.")
    else:
        if isinstance(stats, dict):
            stats_dict = stats
        elif hasattr(stats, "to_dict"):
            stats_dict = stats.to_dict()
        else:
            try:
                stats_dict = dict(stats)
            except Exception:
                stats_dict = {}
        print("\nFit statistics (selected):")
        for k in ("n_params", "n_obs", "df", "gfi", "agfi", "cfi", "tli", "rmsea"):
            if k in stats_dict:
                print(f"  {k}: {stats_dict[k]}")

    # Optionally generate an HTML report (opt-in via --report or env var SEMOPY_DEMO_REPORT=1)
    should_report = args.report or os.environ.get("SEMOPY_DEMO_REPORT", "0") == "1"
    if should_report and sem_report is not None:
        try:
            sem_report(model, data=data, filename="sem_report.html")
            print("\nAn HTML report was generated: sem_report.html")
        except Exception as e:
            print(f"\nReport generation skipped due to: {e}")
    elif should_report and sem_report is None:
        print("\nReport generation not available: semopy.report could not be imported.")


if __name__ == "__main__":
    main()
