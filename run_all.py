#!/usr/bin/env python3
"""
Main entry point for Feature Importance Audit benchmark.
Runs all three experiments or targeted experiments via CLI, generates figures,
and saves metrics to results/metrics.json.
"""

import sys
from pathlib import Path

# Ensure root scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import json
import time

from config import DEFAULT_N_SAMPLES, RANDOM_SEED, RESULTS_DIR
from experiments.exp1_cardinality_bias import run_experiment_1
from experiments.exp2_multicollinearity import run_experiment_2
from experiments.exp3_pruning_pipeline import run_experiment_3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Importance Audit: MDI vs Permutation vs TreeSHAP"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        choices=["all", "1", "2", "3"],
        default="all",
        help="Specify which experiment to run (default: all)"
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic samples (default: {DEFAULT_N_SAMPLES})"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure rendering and saving"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_fig = not args.no_plot

    print("================================================================")
    print("  Feature Importance Audit: MDI vs Permutation vs TreeSHAP")
    print(f"  Configuration: N={args.n_samples}, Seed={args.seed}, SaveFigures={save_fig}")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("================================================================\n")

    t_start = time.time()
    results = {}

    # Run Experiment 1
    if args.experiment in ["all", "1"]:
        r1_df = run_experiment_1(
            n_samples=args.n_samples,
            random_state=args.seed,
            save_fig=save_fig
        )
        results["experiment_1_cardinality_bias"] = r1_df.to_dict(orient="records")

    # Run Experiment 2
    if args.experiment in ["all", "2"]:
        r2_dict = run_experiment_2(
            n_samples=args.n_samples,
            random_state=args.seed,
            save_fig=save_fig
        )
        results["experiment_2_multicollinearity_split"] = r2_dict

    # Run Experiment 3
    if args.experiment in ["all", "3"]:
        r3_dict = run_experiment_3(
            n_samples=args.n_samples,
            random_state=args.seed,
            save_fig=save_fig
        )
        results["experiment_3_pruning_performance"] = r3_dict

    # Save metrics JSON if all experiments were run
    if args.experiment == "all":
        metrics_path = RESULTS_DIR / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nAll experiments completed successfully in {time.time() - t_start:.2f}s!")
        print(f"Full metrics saved to: {metrics_path}")
    else:
        print(f"\nExperiment {args.experiment} completed in {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    main()
