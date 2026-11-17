from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    file_path: Path
    id_col: str
    churn_col: str
    value_col: str
    value_label: str
    description: str
    leakage_cols: List[str] = field(default_factory=list)


DATASETS: Dict[str, DatasetSpec] = {
    "iranian": DatasetSpec(
        key="iranian",
        name="UCI Iranian Telecom Churn",
        file_path=RAW_DIR / "iranian.csv",
        id_col="customer_id",
        churn_col="churn",
        value_col="customer_value",
        value_label="Customer Value Score",
        description=(
            "Primary research dataset (3,150 telecom subscribers) with usage, "
            "complaints, charge level, tariff plan, tenure, and customer value."
        ),
        leakage_cols=[],
    ),
    "ibm_telco": DatasetSpec(
        key="ibm_telco",
        name="IBM Telco Customer Churn",
        file_path=ROOT_DIR / "IBM_Telco_customer_churn_IBM_dataset.csv",
        id_col="CustomerID",
        churn_col="Churn Value",
        value_col="CLTV",
        value_label="CLTV (USD)",
        description=(
            "External benchmark dataset (7,043 subscribers) with contract, "
            "service subscriptions, monthly/total charges, CLTV, and churn."
        ),
        leakage_cols=[
            "Count",
            "Country",
            "State",
            "City",
            "Zip Code",
            "Lat Long",
            "Latitude",
            "Longitude",
            "Churn Label",
            "Churn Score",
            "Churn Reason",
        ],
    ),
}


@dataclass(frozen=True)
class OfferSpec:
    offer_id: str
    name: str
    action_type: str
    cost: float
    expected_retention_lift: float  # Used ONLY in clearly labeled simulation
    min_churn_prob: float
    min_value_quantile: float
    target_segments: List[str]
    description: str


OFFER_CATALOG: List[OfferSpec] = [
    OfferSpec(
        offer_id="VIP_CONCIERGE",
        name="VIP Priority Care & Loyalty Bonus",
        action_type="Dedicated Retention Call + Bonus",
        cost=35.0,
        expected_retention_lift=0.35,
        min_churn_prob=0.45,
        min_value_quantile=0.70,
        target_segments=["High-Value At Risk", "High-Value Loyal"],
        description="High-touch outreach with loyalty credit for high-value at-risk subscribers.",
    ),
    OfferSpec(
        offer_id="CONTRACT_LOCK",
        name="Annual Plan Discount (15% off)",
        action_type="Contract Upgrade Offer",
        cost=22.0,
        expected_retention_lift=0.28,
        min_churn_prob=0.35,
        min_value_quantile=0.40,
        target_segments=["High-Value At Risk", "Growing", "Occasional"],
        description="Targeted contract-lock incentive to stabilize medium/high-value month-to-month users.",
    ),
    OfferSpec(
        offer_id="SERVICE_RECOVERY",
        name="Service Recovery & Technical Credit",
        action_type="Proactive Support + Fee Waiver",
        cost=15.0,
        expected_retention_lift=0.25,
        min_churn_prob=0.30,
        min_value_quantile=0.20,
        target_segments=["High-Value At Risk", "Inactive / Declining", "Occasional"],
        description="Automated service credit for subscribers experiencing call failures or complaints.",
    ),
    OfferSpec(
        offer_id="USAGE_BUNDLE",
        name="Personalized Data & Voice Booster",
        action_type="Next Best Offer (Upsell/Cross-sell)",
        cost=8.0,
        expected_retention_lift=0.14,
        min_churn_prob=0.15,
        min_value_quantile=0.25,
        target_segments=["Growing", "High-Value Loyal", "Occasional"],
        description="Low-cost tailored usage bundle to deepen engagement and ARPU.",
    ),
    OfferSpec(
        offer_id="DIGITAL_NUDGE",
        name="Self-Service Digital Check-In",
        action_type="Low-Cost Automated Nudge",
        cost=2.0,
        expected_retention_lift=0.06,
        min_churn_prob=0.0,
        min_value_quantile=0.0,
        target_segments=["Occasional", "Inactive / Declining", "Growing", "High-Value Loyal", "High-Value At Risk"],
        description="Zero/low-cost automated engagement message when higher-cost offers are not justified.",
    ),
]

