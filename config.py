"""
Global configuration and directory setup for Feature Importance Audit benchmark.
"""

import os
import sys
from pathlib import Path

# Set environment variables for headless and sandboxed execution before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("SKB_DATA_DIRECTORY", "/tmp/skrub_data")
os.environ.setdefault("SCIKIT_LEARN_DATA", "/tmp/scikit_learn_data")

# Path definitions
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add scripts directory to sys.path so modules can import seamlessly
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reproducibility seed and dataset size
RANDOM_SEED = 42
DEFAULT_N_SAMPLES = 3000

# Default LightGBM parameters for benchmark (Split count importance)
LGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "importance_type": "split",
}


def setup_plotting_theme():
    """Apply consistent styling across all figures."""
    import matplotlib.pyplot as plt

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
