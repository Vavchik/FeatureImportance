"""
Synthetic data generation functions for feature importance benchmarks.
All functions have known ground truth to verify feature ranking validity.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def make_cardinality_data(
    n_samples: int = 3000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate dataset for Experiment 1 (Cardinality Bias).

    Includes:
    - 2 continuous informative features ~ N(0, 1)
    - 1 binary informative feature ~ Bernoulli(0.5) with +2.0 impact on target
    - 1 high-cardinality noise feature (user_id: 1000 unique integers)
    - 1 continuous noise feature ~ N(0, 1)
    - 1 low-cardinality binary noise feature ~ Binomial(1, 0.5)

    Ground truth target:
        y = 1.5 * x1 + 1.5 * x2 + 2.0 * x3_binary + eps, eps ~ N(0, 1.0)
    """
    np.random.seed(random_state)

    # True informative continuous features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)

    # Low-cardinality true signal (binary informative flag: 50% positive, +2.0 boost)
    x3_binary = (np.random.uniform(0, 1, n_samples) > 0.5).astype(float)

    # Pure noise features:
    # High cardinality categorical / numeric noise (user_id: 1000 unique integers)
    noise_high_card = np.random.randint(0, 1000, n_samples)
    # Continuous noise
    noise_cont = np.random.normal(0, 1, n_samples)
    # Low cardinality binary noise (coin toss)
    noise_binary = np.random.binomial(1, 0.5, n_samples)

    # Ground truth target: y depends strictly on x1, x2, x3_binary. NO DEPENDENCE on noise!
    y = 1.5 * x1 + 1.5 * x2 + 2.0 * x3_binary + np.random.normal(0, 1.0, n_samples)

    df = pd.DataFrame({
        "signal_cont_1": x1,
        "signal_cont_2": x2,
        "signal_binary_flag": x3_binary,
        "noise_high_card (user_id)": noise_high_card,
        "noise_continuous": noise_cont,
        "noise_binary": noise_binary
    })

    return df, y


def make_multicollinearity_data(
    n_samples: int = 3000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Generate dataset for Experiment 2 (Multicollinearity Splitting).

    Returns:
    - df_no_collin: 3 features [feature_A, feature_B, noise]
    - df_collin: 4 features [feature_A, feature_A_twin, feature_B, noise]
    - y: continuous target

    Ground truth target:
        y = 2.5 * feature_A + 1.5 * feature_B + eps, eps ~ N(0, 0.5)
    """
    np.random.seed(random_state)

    # True informative base
    x_base = np.random.normal(0, 1, n_samples)
    # Collinear twin with correlation r > 0.999
    x_twin = x_base + np.random.normal(0, 0.01, n_samples)

    # Independent informative feature
    x_indep = np.random.normal(0, 1, n_samples)
    # Noise feature
    noise = np.random.normal(0, 1, n_samples)

    # Target depends on x_base and x_indep
    y = 2.5 * x_base + 1.5 * x_indep + np.random.normal(0, 0.5, n_samples)

    df_no_collin = pd.DataFrame({
        "feature_A": x_base,
        "feature_B": x_indep,
        "noise": noise
    })

    df_collin = pd.DataFrame({
        "feature_A": x_base,
        "feature_A_twin": x_twin,
        "feature_B": x_indep,
        "noise": noise
    })

    return df_no_collin, df_collin, y


def make_pruning_data(
    n_samples: int = 3000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate dataset for Experiment 3 (Feature Pruning Pipeline).

    Returns:
    - X: DataFrame with 7 columns (3 signals, 1 collinear clone, 3 noises)
    - y: continuous target

    Ground truth target:
        y = 1.6 * income + 1.6 * credit_score + 2.2 * vip_flag + eps, eps ~ N(0, 0.5)
    """
    np.random.seed(random_state)

    # Core continuous features
    f_score = np.random.normal(0, 1, size=n_samples)
    f_income = np.random.normal(0, 1, size=n_samples)

    # Crucial low-cardinality binary flag (vip client: 25% positive, +2.2 impact)
    f_vip = (np.random.uniform(0, 1, size=n_samples) > 0.75).astype(float)

    # Collinear clone of income (correlation > 0.99)
    f_income_clone = f_income + np.random.normal(0, 0.02, size=n_samples)

    # 3 Pure noise features:
    # High cardinality pseudo client ID
    f_client_id = np.random.randint(1000, 9999, size=n_samples)
    # Continuous normal noise
    f_noise_cont = np.random.normal(0, 1, size=n_samples)
    # Uniform noise
    f_noise_unif = np.random.uniform(-1, 1, size=n_samples)

    # Ground truth non-linear target
    y = 1.6 * f_income + 1.6 * f_score + 2.2 * f_vip + np.random.normal(0, 0.5, size=n_samples)

    X = pd.DataFrame({
        "credit_score": f_score,
        "income": f_income,
        "vip_flag (binary)": f_vip,
        "income_clone": f_income_clone,
        "noise_uniform": f_noise_unif,
        "noise_continuous": f_noise_cont,
        "client_id (noise)": f_client_id
    })

    return X, y
