"""
One-command orchestration for the consolidated Telco churn project.

Run:
    python -m customer_churn.run_all

Stages:
    1. baseline training
    2. hyperparameter tuning
    3. threshold optimization

Threshold optimization always prefers the tuned artifact when tuning
successfully completes.
"""

from __future__ import annotations

import subprocess
import sys


def run(module: str) -> None:
    print("\n" + "=" * 80)
    print(f"RUNNING: {module}")
    print("=" * 80)
    result = subprocess.run(
        [sys.executable, "-m", module],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{module} failed with exit code {result.returncode}"
        )


def main() -> None:
    run("customer_churn.train")
    run("customer_churn.tune")
    run("customer_churn.threshold")

    print("\n" + "=" * 80)
    print("COMPLETE CONSOLIDATED PIPELINE FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()
