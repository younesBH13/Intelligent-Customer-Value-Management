from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .data_prep import PreparedDataset


@dataclass
class ChurnBenchmarkResult:
    best_model_name: str
    best_model: Any
    metrics_table: List[Dict[str, Any]]
    predicted_probs: np.ndarray
    test_y: np.ndarray
    test_probs: np.ndarray
    confusion_matrix: List[List[int]]
    roc_curves: Dict[str, Dict[str, List[float]]]
    pr_curves: Dict[str, Dict[str, List[float]]]
    calibration_curves: Dict[str, Dict[str, List[float]]]


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def run_churn_benchmark(prepared: PreparedDataset, test_size: float = 0.2) -> ChurnBenchmarkResult:
    df = prepared.df_clean
    X = df[prepared.feature_cols]
    y = df[prepared.churn_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    pos_count = max(int(y_train.sum()), 1)
    neg_count = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = neg_count / pos_count

    candidates: Dict[str, Any] = {
        "Logistic Regression (Baseline)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics_table: List[Dict[str, Any]] = []
    trained: Dict[str, Any] = {}
    test_probs_map: Dict[str, np.ndarray] = {}
    roc_curves: Dict[str, Dict[str, List[float]]] = {}
    pr_curves: Dict[str, Dict[str, List[float]]] = {}
    calibration_curves: Dict[str, Dict[str, List[float]]] = {}

    best_name = "Logistic Regression (Baseline)"
    best_pr_auc = -1.0

    for name, model in candidates.items():
        # 5-fold Stratified CV on training split
        cv_res = cross_validate(
            clone(model),
            X_train,
            y_train,
            cv=skf,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        cv_pr_mean = float(np.mean(cv_res["test_pr_auc"]))
        cv_pr_std = float(np.std(cv_res["test_pr_auc"]))
        cv_roc_mean = float(np.mean(cv_res["test_roc_auc"]))

        # Fit on full training split and evaluate on holdout test split
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        pr_auc = float(average_precision_score(y_test, probs))
        roc_auc = float(roc_auc_score(y_test, probs))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        brier = float(brier_score_loss(y_test, probs))
        ece = _expected_calibration_error(y_test, probs, n_bins=10)

        metrics_table.append(
            {
                "model": name,
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "cv5_pr_auc": f"{cv_pr_mean:.4f} ± {cv_pr_std:.4f}",
                "cv5_roc_auc": round(cv_roc_mean, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "brier_score": round(brier, 4),
                "ece": round(ece, 4),
            }
        )

        fpr, tpr, _ = roc_curve(y_test, probs)
        p_curve, r_curve, _ = precision_recall_curve(y_test, probs)
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")

        roc_curves[name] = {
            "fpr": [round(float(x), 4) for x in fpr[:: max(1, len(fpr) // 120)]],
            "tpr": [round(float(x), 4) for x in tpr[:: max(1, len(tpr) // 120)]],
        }
        pr_curves[name] = {
            "precision": [round(float(x), 4) for x in p_curve[:: max(1, len(p_curve) // 120)]],
            "recall": [round(float(x), 4) for x in r_curve[:: max(1, len(r_curve) // 120)]],
        }
        calibration_curves[name] = {
            "prob_pred": [round(float(x), 4) for x in prob_pred],
            "prob_true": [round(float(x), 4) for x in prob_true],
        }

        trained[name] = model
        test_probs_map[name] = probs

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name

    best_model = trained[best_name]
    best_test_probs = test_probs_map[best_name]
    cm = confusion_matrix(y_test, (best_test_probs >= 0.5).astype(int)).tolist()
    full_probs = best_model.predict_proba(X)[:, 1]

    return ChurnBenchmarkResult(
        best_model_name=best_name,
        best_model=best_model,
        metrics_table=metrics_table,
        predicted_probs=full_probs,
        test_y=y_test,
        test_probs=best_test_probs,
        confusion_matrix=cm,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
        calibration_curves=calibration_curves,
    )
