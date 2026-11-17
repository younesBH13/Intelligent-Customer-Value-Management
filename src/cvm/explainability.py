from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _extract_shap_values(
    model: Any,
    X: pd.DataFrame,
    background: pd.DataFrame | None = None,
) -> np.ndarray:
    """Compute SHAP values for tree-based, linear, or fallback models."""
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler")
        estimator = list(model.named_steps.values())[-1]
        X_trans = scaler.transform(X) if scaler is not None else X.values
        explainer = shap.LinearExplainer(estimator, X_trans)
        vals = explainer.shap_values(X_trans)
    else:
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X)
        except Exception:
            # Fallback for XGBoost >=3.x base_score bracket formatting in TreeExplainer
            bg = background if background is not None else X
            bg_sample = bg.iloc[: min(len(bg), 50)]
            pred_fn = (
                (lambda data: model.predict_proba(data)[:, 1])
                if hasattr(model, "predict_proba")
                else model.predict
            )
            explainer = shap.Explainer(pred_fn, bg_sample)
            vals = explainer(X).values

    arr = np.asarray(vals)
    # Binary classification in some SHAP versions returns (n, f, 2) or list of 2 arrays
    if isinstance(vals, list):
        arr = np.asarray(vals[1] if len(vals) == 2 else vals[0])
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    return arr


def compute_global_importance(
    model: Any,
    X: pd.DataFrame,
    feature_cols: List[str],
    max_samples: int = 400,
) -> List[Dict[str, Any]]:
    X_sample = X[feature_cols].iloc[: min(len(X), max_samples)]
    shap_vals = _extract_shap_values(model, X_sample)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    items = [
        {"feature": feat, "mean_abs_shap": round(float(val), 5)}
        for feat, val in zip(feature_cols, mean_abs)
    ]
    items.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    return items


def explain_single_customer(
    model: Any,
    row_df: pd.DataFrame,
    feature_cols: List[str],
    top_k: int = 8,
    background_df: pd.DataFrame | None = None,
) -> List[Dict[str, Any]]:
    X_one = row_df[feature_cols]
    bg = background_df[feature_cols] if background_df is not None else None
    shap_vals = _extract_shap_values(model, X_one, background=bg)[0]

    items = [
        {
            "feature": feat,
            "feature_value": round(float(X_one.iloc[0][feat]), 4),
            "shap_contribution": round(float(val), 5),
        }
        for feat, val in zip(feature_cols, shap_vals)
    ]
    items.sort(key=lambda d: abs(d["shap_contribution"]), reverse=True)
    return items[:top_k]
