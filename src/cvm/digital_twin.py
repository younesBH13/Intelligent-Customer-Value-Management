from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, OFFER_CATALOG
from .recommender import _is_eligible_for_offer


def _recompute_derived_features(row: pd.Series, dataset_key: str) -> pd.Series:
    """Keep engineered ratios consistent when raw levers are modified in the Digital Twin."""
    r = row.copy()
    if dataset_key == "iranian":
        sub_len = max(float(r.get("subscription_length", 1)), 1.0)
        freq_use = max(float(r.get("frequency_of_use", 1)), 1.0)
        r["calls_per_month"] = float(r.get("frequency_of_use", 0)) / sub_len
        r["sms_per_month"] = float(r.get("frequency_of_sms", 0)) / sub_len
        r["seconds_per_month"] = float(r.get("seconds_of_use", 0)) / sub_len
        r["avg_call_duration"] = float(r.get("seconds_of_use", 0)) / freq_use
        r["failure_rate"] = float(r.get("call_failure", 0)) / freq_use
        r["revenue_proxy"] = float(r.get("charge_amount", 0)) * sub_len
    elif dataset_key == "ibm_telco":
        tenure = max(float(r.get("tenure_months", 1)), 1.0)
        monthly = float(r.get("monthly_charges", 0))
        if "total_charges" not in r or pd.isna(r["total_charges"]):
            r["total_charges"] = monthly * tenure
        r["avg_monthly_spend"] = float(r["total_charges"]) / tenure
        r["charge_tenure_ratio"] = monthly / tenure
        r["is_long_term_contract"] = int(
            (r.get("contract_one_year", 0) == 1) or (r.get("contract_two_year", 0) == 1)
        )
        r["has_security_bundle"] = int(
            (r.get("online_security_yes", 0) == 1) and (r.get("tech_support_yes", 0) == 1)
        )
    return r


def simulate_digital_twin(
    dataset_key: str,
    overrides: Dict[str, Any],
    base_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run live Churn + CLV prediction and Next Best Offer evaluation for a
    new customer profile or a modified Customer Digital Twin.
    """
    out_dir = ARTIFACTS_DIR / dataset_key
    with open(out_dir / "report.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    df = pd.read_csv(out_dir / "scored_customers.csv")
    churn_model = joblib.load(out_dir / "churn_model.joblib")
    clv_model = joblib.load(out_dir / "clv_model.joblib")

    id_col = report["dataset"]["id_col"]
    feature_cols = report["dataset"]["feature_cols"]

    if base_customer_id:
        match = df[df[id_col].astype(str) == str(base_customer_id)]
        if match.empty:
            raise ValueError(f"Customer '{base_customer_id}' not found in '{dataset_key}'")
        base_row = match.iloc[0].copy()
        baseline = {
            "customer_id": str(base_row[id_col]),
            "segment_name": str(base_row["segment_name"]),
            "churn_prob": float(base_row["churn_prob"]),
            "predicted_clv": float(base_row["predicted_clv"]),
            "priority_score": float(base_row["priority_score"]),
            "recommended_offer": str(base_row["recommended_offer"]),
        }
    else:
        # Median synthetic baseline for brand-new customer prediction
        base_row = df[feature_cols].median(numeric_only=True)
        base_row["segment_name"] = "Growing"
        baseline = None

    twin_row = base_row.copy()
    for k, v in overrides.items():
        if k in twin_row.index or k in feature_cols:
            twin_row[k] = v

    # Handle contract switching cleanly for ibm_telco if contract_type is passed
    if dataset_key == "ibm_telco" and "contract_type" in overrides:
        ctype = str(overrides["contract_type"]).lower()
        twin_row["contract_month_to_month"] = int("month" in ctype)
        twin_row["contract_one_year"] = int("one" in ctype or "1" in ctype)
        twin_row["contract_two_year"] = int("two" in ctype or "2" in ctype)

    twin_row = _recompute_derived_features(twin_row, dataset_key)
    X_twin = pd.DataFrame([twin_row[feature_cols].to_dict()])[feature_cols]

    new_churn_prob = float(np.clip(churn_model.predict_proba(X_twin)[:, 1][0], 0.0, 1.0))
    new_clv = float(max(0.0, clv_model.predict(X_twin)[0]))
    new_priority = round(new_churn_prob * new_clv, 2)

    # Value quantile relative to existing portfolio
    val_q = float((df["predicted_clv"] <= new_clv).mean())
    seg_name = str(twin_row.get("segment_name", "Growing"))

    best_offer = OFFER_CATALOG[-1]
    best_net = -1e9
    for offer in OFFER_CATALOG:
        if _is_eligible_for_offer(offer, new_churn_prob, val_q, seg_name, twin_row, dataset_key):
            saved = new_churn_prob * new_clv * offer.expected_retention_lift
            net = saved - offer.cost
            if net > best_net:
                best_net = net
                best_offer = offer

    sim_saved = round(new_churn_prob * new_clv * best_offer.expected_retention_lift, 2)
    sim_net = round(sim_saved - best_offer.cost, 2)

    simulated = {
        "churn_prob": round(new_churn_prob, 4),
        "predicted_clv": round(new_clv, 2),
        "priority_score": new_priority,
        "recommended_offer": best_offer.name,
        "recommended_action": best_offer.action_type,
        "offer_cost": best_offer.cost,
        "sim_expected_net_benefit": sim_net,
    }

    deltas = None
    if baseline is not None:
        deltas = {
            "delta_churn_prob": round(simulated["churn_prob"] - baseline["churn_prob"], 4),
            "delta_clv": round(simulated["predicted_clv"] - baseline["predicted_clv"], 2),
            "delta_priority": round(simulated["priority_score"] - baseline["priority_score"], 2),
        }

    return {
        "dataset_key": dataset_key,
        "base_customer_id": base_customer_id,
        "applied_overrides": overrides,
        "baseline": baseline,
        "simulated": simulated,
        "deltas": deltas,
    }

