from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DATASETS, DatasetSpec


@dataclass
class PreparedDataset:
    key: str
    spec: DatasetSpec
    df_clean: pd.DataFrame
    feature_cols: List[str]
    rfm_cols: List[str]
    id_col: str
    churn_col: str
    value_col: str
    quality_report: Dict[str, Any]


def _snake_case(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    return s.strip("_").lower()


def _build_quality_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    churn_col: str,
    value_col: str,
    dropped_cols: List[str],
    notes: List[str],
) -> Dict[str, Any]:
    missing_raw = {col: int(cnt) for col, cnt in df_raw.isna().sum().items() if cnt > 0}
    missing_clean = {col: int(cnt) for col, cnt in df_clean.isna().sum().items() if cnt > 0}
    churn_series = df_clean[churn_col]
    value_series = df_clean[value_col]

    return {
        "raw_rows": int(len(df_raw)),
        "raw_cols": int(df_raw.shape[1]),
        "clean_rows": int(len(df_clean)),
        "clean_cols": int(df_clean.shape[1]),
        "duplicate_rows_raw": int(df_raw.duplicated().sum()),
        "missing_raw": missing_raw,
        "missing_clean": missing_clean,
        "churn_rate": float(churn_series.mean()),
        "churn_positive_count": int(churn_series.sum()),
        "churn_negative_count": int((1 - churn_series).sum()),
        "value_summary": {
            "mean": float(value_series.mean()),
            "median": float(value_series.median()),
            "std": float(value_series.std()),
            "min": float(value_series.min()),
            "max": float(value_series.max()),
        },
        "dropped_leakage_or_pii_cols": dropped_cols,
        "notes": notes,
    }


def _prepare_iranian(spec: DatasetSpec) -> PreparedDataset:
    df_raw = pd.read_csv(spec.file_path)
    df = df_raw.copy()
    df.columns = [_snake_case(c) for c in df.columns]

    # Add pseudonymous ID since UCI Iranian dataset has no explicit customer ID
    df.insert(0, "customer_id", [f"IRN-{i+1:04d}" for i in range(len(df))])

    # Ensure numeric types
    for col in df.columns:
        if col != "customer_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering (using only pre-churn behavioral & account attributes)
    sub_len = df["subscription_length"].replace(0, 1)
    freq_use = df["frequency_of_use"].replace(0, 1)

    df["calls_per_month"] = df["frequency_of_use"] / sub_len
    df["sms_per_month"] = df["frequency_of_sms"] / sub_len
    df["seconds_per_month"] = df["seconds_of_use"] / sub_len
    df["avg_call_duration"] = df["seconds_of_use"] / freq_use
    df["failure_rate"] = df["call_failure"] / freq_use
    df["revenue_proxy"] = df["charge_amount"] * sub_len

    # Binary complaint / active flags for business rules & segmentation
    df["has_complaint"] = (df["complains"] > 0).astype(int)
    df["is_active_status"] = (df["status"] == 1).astype(int)

    feature_cols = [
        "call_failure",
        "complains",
        "subscription_length",
        "charge_amount",
        "seconds_of_use",
        "frequency_of_use",
        "frequency_of_sms",
        "distinct_called_numbers",
        "age_group",
        "tariff_plan",
        "status",
        "age",
        "calls_per_month",
        "sms_per_month",
        "seconds_per_month",
        "avg_call_duration",
        "failure_rate",
        "revenue_proxy",
    ]

    rfm_cols = [
        "subscription_length",  # Tenure / Recency-activity proxy
        "frequency_of_use",     # Frequency
        "charge_amount",        # Monetary tier
        "seconds_of_use",       # Consumption intensity
        "customer_value",       # Value indicator
    ]

    notes = [
        "Normalized column names (removed double spaces).",
        "Generated pseudonymous customer_id (IRN-0001..IRN-3150).",
        "Excluded customer_value and churn from predictive feature_cols to prevent target leakage across Stage 3 & Stage 4.",
        "Engineered rate/intensity features: calls_per_month, sms_per_month, seconds_per_month, avg_call_duration, failure_rate, revenue_proxy.",
    ]

    quality_report = _build_quality_report(
        df_raw=df_raw,
        df_clean=df,
        churn_col="churn",
        value_col="customer_value",
        dropped_cols=[],
        notes=notes,
    )

    return PreparedDataset(
        key=spec.key,
        spec=spec,
        df_clean=df,
        feature_cols=feature_cols,
        rfm_cols=rfm_cols,
        id_col="customer_id",
        churn_col="churn",
        value_col="customer_value",
        quality_report=quality_report,
    )


def _prepare_ibm_telco(spec: DatasetSpec) -> PreparedDataset:
    df_raw = pd.read_csv(spec.file_path)
    df = df_raw.copy()

    # Coerce Total Charges (blank strings exist for tenure == 0)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    blank_total = int(df["Total Charges"].isna().sum())
    df["Total Charges"] = df["Total Charges"].fillna(df["Monthly Charges"] * df["Tenure Months"])

    # Drop PII/geo/post-hoc leakage columns from modeling, keep CustomerID for tracking
    dropped_cols = [c for c in spec.leakage_cols if c in df.columns]
    df = df.drop(columns=dropped_cols)

    # Rename columns to clean snake_case
    df.columns = [_snake_case(c) for c in df.columns]

    # Binary encode Yes/No and Gender
    yes_no_cols = [
        "senior_citizen",
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
    ]
    for col in yes_no_cols:
        df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    df["gender_male"] = (df["gender"].astype(str).str.strip().str.lower() == "male").astype(int)
    df = df.drop(columns=["gender"])

    # Multi-category service columns -> binary 'Yes' indicator + preserve one-hot
    service_addon_cols = [
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    for col in service_addon_cols:
        df[f"{col}_yes"] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    df["service_count"] = df[[f"{c}_yes" for c in service_addon_cols]].sum(axis=1) + df["phone_service"]
    df["has_streaming"] = ((df["streaming_tv_yes"] == 1) | (df["streaming_movies_yes"] == 1)).astype(int)
    df["has_security_bundle"] = ((df["online_security_yes"] == 1) & (df["tech_support_yes"] == 1)).astype(int)
    df["has_fiber"] = (df["internet_service"].astype(str).str.strip().str.lower() == "fiber optic").astype(int)
    df["has_internet"] = (df["internet_service"].astype(str).str.strip().str.lower() != "no").astype(int)
    df["is_long_term_contract"] = df["contract"].isin(["One year", "Two year"]).astype(int)
    df["is_auto_pay"] = df["payment_method"].astype(str).str.contains("automatic", case=False, na=False).astype(int)

    tenure_safe = df["tenure_months"].replace(0, 1)
    df["avg_monthly_spend"] = df["total_charges"] / tenure_safe
    df["charge_tenure_ratio"] = df["monthly_charges"] / tenure_safe

    # One-hot encode remaining categorical columns
    cat_cols = ["internet_service", "contract", "payment_method"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    # Clean newly created dummy column names
    df.columns = [_snake_case(c) for c in df.columns]

    # Drop raw string addon columns now that _yes + service_count exist
    df = df.drop(columns=service_addon_cols)

    excluded = {"customerid", "churn_value", "cltv"}
    feature_cols = [c for c in df.columns if c not in excluded]

    rfm_cols = [
        "tenure_months",
        "service_count",
        "monthly_charges",
        "total_charges",
        "cltv",
    ]

    notes = [
        f"Coerced Total Charges to numeric; imputed {blank_total} blank entries using Monthly Charges * Tenure Months.",
        "Dropped post-hoc leakage columns (Churn Score, Churn Reason, Churn Label) and geo/constant columns.",
        "Excluded CLTV and churn_value from predictive feature_cols to prevent target leakage.",
        "Engineered service_count, avg_monthly_spend, charge_tenure_ratio, has_streaming, has_security_bundle, is_long_term_contract, is_auto_pay.",
    ]

    quality_report = _build_quality_report(
        df_raw=df_raw,
        df_clean=df,
        churn_col="churn_value",
        value_col="cltv",
        dropped_cols=dropped_cols,
        notes=notes,
    )

    return PreparedDataset(
        key=spec.key,
        spec=spec,
        df_clean=df,
        feature_cols=feature_cols,
        rfm_cols=rfm_cols,
        id_col="customerid",
        churn_col="churn_value",
        value_col="cltv",
        quality_report=quality_report,
    )


def load_and_prepare(dataset_key: str) -> PreparedDataset:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {list(DATASETS.keys())}")
    spec = DATASETS[dataset_key]
    if dataset_key == "iranian":
        return _prepare_iranian(spec)
    return _prepare_ibm_telco(spec)

