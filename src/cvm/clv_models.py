from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import RANDOM_STATE
from .data_prep import PreparedDataset


@dataclass
class CLVBenchmarkResult:
    best_model_name: str
    best_model: Any
    metrics_table: List[Dict[str, Any]]
    predicted_values: np.ndarray
    test_actual: np.ndarray
    test_pred: np.ndarray


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    nonzero_mask = np.abs(y_true) > 1e-6
    if not nonzero_mask.any():
        return None
    return float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0
    )


def run_clv_benchmark(prepared: PreparedDataset, test_size: float = 0.2) -> CLVBenchmarkResult:
    df = prepared.df_clean
    X = df[prepared.feature_cols]
    y = df[prepared.value_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    candidates: Dict[str, Any] = {
        "Linear Regression (Baseline)": Pipeline(
            [("scaler", StandardScaler()), ("reg", LinearRegression())]
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM Regressor": LGBMRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    metrics_table: List[Dict[str, Any]] = []
    trained: Dict[str, Any] = {}
    test_preds: Dict[str, np.ndarray] = {}

    best_name = "Linear Regression (Baseline)"
    best_rmse = float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), a_min=0.0, a_max=None)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        mape = _safe_mape(y_test, preds)

        metrics_table.append(
            {
                "model": name,
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 4),
                "mape_pct": round(mape, 2) if mape is not None else None,
            }
        )
        trained[name] = model
        test_preds[name] = preds

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    best_model = trained[best_name]
    full_preds = np.clip(best_model.predict(X), a_min=0.0, a_max=None)

    return CLVBenchmarkResult(
        best_model_name=best_name,
        best_model=best_model,
        metrics_table=metrics_table,
        predicted_values=full_preds,
        test_actual=y_test,
        test_pred=test_preds[best_name],
    )

