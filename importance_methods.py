"""
Wrappers and normalization routines for feature importance algorithms:
- MDI (Mean Decrease in Impurity / Split count)
- Permutation Importance
- TreeSHAP (|SHAP| mean magnitude)
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import shap


def compute_mdi(model) -> np.ndarray:
    """
    Extract default MDI / split count importances from a trained tree/boosting model.
    """
    return np.array(model.feature_importances_, dtype=float)


def compute_permutation(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Compute Permutation Feature Importance (drop in performance metric).
    Returns mean importance per feature.
    """
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return perm.importances_mean


def compute_treeshap(
    model,
    X: Union[pd.DataFrame, np.ndarray]
) -> np.ndarray:
    """
    Compute mean absolute TreeSHAP values for each feature on dataset X.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return np.mean(np.abs(shap_values), axis=0)


def normalize_to_percentage(
    importances: np.ndarray,
    clip_negative: bool = False
) -> np.ndarray:
    """
    Normalize importance scores to percentages summing to 100%.

    If clip_negative=True, clips scores at 0 before normalization
    (standard practice for permutation importance where noise can give negative scores).
    """
    vals = np.maximum(0.0, importances) if clip_negative else np.array(importances, dtype=float)
    total = np.sum(vals)
    if total <= 0:
        return np.zeros_like(vals)
    return (vals / total) * 100.0
