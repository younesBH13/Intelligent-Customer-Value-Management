from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE
from .data_prep import PreparedDataset


def _compute_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute cumulative uplift curve & AUUC (sorted by predicted CATE descending)."""
    df_u = pd.DataFrame({"y": y_true, "w": treatment, "tau_hat": cate_pred})
    df_u = df_u.sort_values("tau_hat", ascending=False).reset_index(drop=True)

    fractions: List[float] = [0.0]
    cum_uplift: List[float] = [0.0]
    n = len(df_u)

    for b in range(1, n_bins + 1):
        k = max(1, int(n * (b / n_bins)))
        sub = df_u.iloc[:k]
        t_mask = sub["w"] == 1
        c_mask = sub["w"] == 0
        # Retention outcome = 1 - churn; uplift = P(retain|W=1) - P(retain|W=0)
        p_t = float((1.0 - sub.loc[t_mask, "y"]).mean()) if t_mask.any() else 0.0
        p_c = float((1.0 - sub.loc[c_mask, "y"]).mean()) if c_mask.any() else 0.0
        uplift_rate = p_t - p_c
        fractions.append(round(b / n_bins, 2))
        cum_uplift.append(round(uplift_rate * (b / n_bins), 4))

    # Area under uplift curve (trapezoidal)
    auuc = float(np.trapz(cum_uplift, fractions))
    # Uplift at top 30%
    top30_idx = fractions.index(0.3) if 0.3 in fractions else 3
    uplift_at_30 = round(cum_uplift[top30_idx] / 0.3, 4) if cum_uplift[top30_idx] != 0 else 0.0

    return {
        "fractions": fractions,
        "cumulative_uplift": cum_uplift,
        "auuc": round(auuc, 5),
        "uplift_at_30pct": uplift_at_30,
    }


def run_simulated_uplift(
    prepared: PreparedDataset,
    churn_probs: np.ndarray,
) -> Dict[str, Any]:
    """
    Semi-synthetic Uplift Modeling (S-Learner, T-Learner, X-Learner).
    IMPORTANT: Public datasets have no historical campaign treatment/response logs.
    Treatment assignment W and retention outcome under treatment are simulated
    for methodological demonstration only.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    X = prepared.df_clean[prepared.feature_cols].copy()
    y_obs_churn = prepared.df_clean[prepared.churn_col].values.astype(int)

    # Simulate realistic campaign treatment W ~ Bernoulli(0.5)
    w = rng.binomial(1, 0.5, size=len(X))

    # True latent heterogeneous retention lift tau(x):
    # Highest for medium/high churn-risk customers (0.25..0.75) who are persuadable,
    # lower for already-loyal (<0.15) or lost-cause (>0.85) customers.
    persuadable = np.exp(-((churn_probs - 0.50) ** 2) / (2 * 0.22 ** 2))
    true_tau = 0.04 + 0.28 * persuadable + rng.normal(0.0, 0.02, size=len(X))
    true_tau = np.clip(true_tau, 0.0, 0.45)

    # Simulate observed churn under treatment W:
    # If W=1 and customer would have churned (y_obs_churn=1), they are retained with prob true_tau
    y_sim_churn = y_obs_churn.copy()
    flip_mask = (w == 1) & (y_obs_churn == 1) & (rng.random(len(X)) < true_tau)
    y_sim_churn[flip_mask] = 0

    X_train, X_test, w_train, w_test, y_train, y_test, tau_train, tau_test = train_test_split(
        X, w, y_sim_churn, true_tau, test_size=0.25, stratify=w, random_state=RANDOM_STATE
    )

    # Target for learners: retention = 1 - churn
    r_train = 1 - y_train

    # 1. S-Learner
    X_s_train = X_train.copy()
    X_s_train["_treatment"] = w_train
    s_model = LGBMClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE, verbose=-1)
    s_model.fit(X_s_train, r_train)

    X_s1 = X_test.copy()
    X_s1["_treatment"] = 1
    X_s0 = X_test.copy()
    X_s0["_treatment"] = 0
    cate_s = s_model.predict_proba(X_s1)[:, 1] - s_model.predict_proba(X_s0)[:, 1]

    # 2. T-Learner
    t1 = LGBMClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE, verbose=-1)
    t0 = LGBMClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE, verbose=-1)
    t1.fit(X_train[w_train == 1], r_train[w_train == 1])
    t0.fit(X_train[w_train == 0], r_train[w_train == 0])
    cate_t = t1.predict_proba(X_test)[:, 1] - t0.predict_proba(X_test)[:, 1]

    # 3. X-Learner (using T-learner base models + imputed treatment effects)
    d1 = r_train[w_train == 1] - t0.predict_proba(X_train[w_train == 1])[:, 1]
    d0 = t1.predict_proba(X_train[w_train == 0])[:, 1] - r_train[w_train == 0]
    tau1_reg = LGBMRegressor(n_estimators=120, max_depth=4, random_state=RANDOM_STATE, verbose=-1)
    tau0_reg = LGBMRegressor(n_estimators=120, max_depth=4, random_state=RANDOM_STATE, verbose=-1)
    tau1_reg.fit(X_train[w_train == 1], d1)
    tau0_reg.fit(X_train[w_train == 0], d0)
    # Propensity e(x) = 0.5 in randomized simulation
    cate_x = 0.5 * tau0_reg.predict(X_test) + 0.5 * tau1_reg.predict(X_test)

    learners = {
        "S-Learner (LightGBM)": cate_s,
        "T-Learner (LightGBM)": cate_t,
        "X-Learner (LightGBM)": cate_x,
    }

    metrics_table: List[Dict[str, Any]] = []
    curves: Dict[str, Any] = {}
    best_name = "T-Learner (LightGBM)"
    best_auuc = -1e9

    for name, preds in learners.items():
        curve = _compute_uplift_curve(y_test, w_test, preds, n_bins=10)
        pehe = float(np.sqrt(np.mean((tau_test - preds) ** 2)))
        metrics_table.append(
            {
                "learner": name,
                "auuc": curve["auuc"],
                "uplift_at_30pct": curve["uplift_at_30pct"],
                "pehe_rmse": round(pehe, 4),
            }
        )
        curves[name] = curve
        if curve["auuc"] > best_auuc:
            best_auuc = curve["auuc"]
            best_name = name

    return {
        "is_simulated": True,
        "note": (
            "SIMULATED UPLIFT EXPERIMENT: No historical campaign treatment/response data exists "
            "in this dataset. Treatment assignment W and retention lift are semi-synthetic "
            "to benchmark S-Learner, T-Learner, and X-Learner."
        ),
        "best_learner": best_name,
        "metrics_table": metrics_table,
        "uplift_curves": curves,
    }

