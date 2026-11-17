from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cvm.agent import ask_cvm_agent  # noqa: E402
from cvm.config import ARTIFACTS_DIR, DATASETS  # noqa: E402
from cvm.digital_twin import simulate_digital_twin  # noqa: E402
from cvm.explainability import explain_single_customer  # noqa: E402


class PredictTwinRequest(BaseModel):
    base_customer_id: Optional[str] = Field(
        None, description="Optional existing customer ID to clone as a Digital Twin."
    )
    overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Feature values or what-if levers to apply."
    )


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question for the CVM Copilot.")

app = FastAPI(
    title="Intelligent CVM Decision-Support API",
    description=(
        "REST API for Customer Value Management: segmentation, CLV estimation, "
        "churn prediction, value-aware Next Best Offer recommendations, and SHAP explanations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_report(dataset_key: str) -> Dict[str, Any]:
    if dataset_key not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_key}'")
    path = ARTIFACTS_DIR / dataset_key / "report.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifacts for '{dataset_key}' not found. Run `python -m cvm.pipeline --dataset {dataset_key}` first.",
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_scored_df(dataset_key: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / dataset_key / "scored_customers.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Scored customers not found for '{dataset_key}'")
    return pd.read_csv(path)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets() -> List[Dict[str, Any]]:
    items = []
    for key, spec in DATASETS.items():
        ready = (ARTIFACTS_DIR / key / "report.json").exists()
        items.append(
            {
                "key": key,
                "name": spec.name,
                "description": spec.description,
                "ready": ready,
            }
        )
    return items


@app.get("/report/{dataset_key}")
def get_report(dataset_key: str) -> Dict[str, Any]:
    return _load_report(dataset_key)


@app.get("/customers/{dataset_key}")
def list_customers(
    dataset_key: str,
    segment: Optional[str] = Query(None),
    risk_tier: Optional[str] = Query(None),
    value_tier: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    report = _load_report(dataset_key)
    df = _load_scored_df(dataset_key)

    if segment:
        df = df[df["segment_name"] == segment]
    if risk_tier:
        df = df[df["risk_tier"] == risk_tier]
    if value_tier:
        df = df[df["value_tier"] == value_tier]

    total = int(len(df))
    page = df.iloc[offset : offset + limit]
    id_col = report["dataset"]["id_col"]

    cols = [
        "priority_rank",
        id_col,
        "segment_name",
        "churn_prob",
        "predicted_clv",
        "priority_score",
        "risk_tier",
        "value_tier",
        "recommended_offer",
        "recommended_action",
        "offer_cost",
        "sim_expected_net_benefit",
    ]
    cols = [c for c in cols if c in page.columns]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": page[cols].to_dict(orient="records"),
    }


@app.get("/explain/{dataset_key}/{customer_id}")
def explain_customer(dataset_key: str, customer_id: str) -> Dict[str, Any]:
    report = _load_report(dataset_key)
    df = _load_scored_df(dataset_key)
    id_col = report["dataset"]["id_col"]
    feature_cols = report["dataset"]["feature_cols"]

    match = df[df[id_col].astype(str) == str(customer_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found in '{dataset_key}'")

    row_df = match.iloc[[0]]
    churn_model = joblib.load(ARTIFACTS_DIR / dataset_key / "churn_model.joblib")
    clv_model = joblib.load(ARTIFACTS_DIR / dataset_key / "clv_model.joblib")

    churn_contrib = explain_single_customer(churn_model, row_df, feature_cols, top_k=8, background_df=df)
    clv_contrib = explain_single_customer(clv_model, row_df, feature_cols, top_k=8, background_df=df)

    r = row_df.iloc[0]
    return {
        "customer_id": str(r[id_col]),
        "segment_name": str(r["segment_name"]),
        "churn_prob": float(r["churn_prob"]),
        "predicted_clv": float(r["predicted_clv"]),
        "priority_score": float(r["priority_score"]),
        "recommended_offer": str(r["recommended_offer"]),
        "recommended_action": str(r["recommended_action"]),
        "recommendation_rationale": str(r["recommendation_rationale"]),
        "churn_shap_contributions": churn_contrib,
        "clv_shap_contributions": clv_contrib,
        "disclaimer": report["stage6_explainability"]["note"],
    }


@app.get("/export/{dataset_key}/aggregate")
def export_aggregate(dataset_key: str) -> Dict[str, Any]:
    report = _load_report(dataset_key)
    return {
        "dataset": report["dataset"]["name"],
        "data_quality": report["stage1_data_quality"],
        "cluster_profiles": report["stage2_segmentation"]["cluster_profiles"],
        "offer_distribution": report["stage5_recommendations"]["offer_distribution"],
        "policy_comparison": report["stage5_recommendations"]["policy_comparison"],
    }


@app.get("/uplift/{dataset_key}")
def get_uplift_simulation(dataset_key: str) -> Dict[str, Any]:
    report = _load_report(dataset_key)
    return report.get("stage5b_uplift_simulation", {})


@app.get("/nlp/{dataset_key}")
def get_nlp_insights(dataset_key: str) -> Dict[str, Any]:
    report = _load_report(dataset_key)
    return report.get("optional_nlp_insights", {})


@app.post("/predict/{dataset_key}")
def predict_or_simulate_twin(dataset_key: str, req: PredictTwinRequest) -> Dict[str, Any]:
    _load_report(dataset_key)
    try:
        return simulate_digital_twin(
            dataset_key=dataset_key,
            overrides=req.overrides,
            base_customer_id=req.base_customer_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agent/{dataset_key}/ask")
def ask_agent(dataset_key: str, req: AgentQueryRequest) -> Dict[str, Any]:
    _load_report(dataset_key)
    return ask_cvm_agent(dataset_key=dataset_key, query=req.query)


@app.get("/portfolio-sample/{dataset_key}")
def get_portfolio_sample(dataset_key: str, n: int = Query(400, ge=50, le=1500)) -> List[Dict[str, Any]]:
    report = _load_report(dataset_key)
    df = _load_scored_df(dataset_key)
    id_col = report["dataset"]["id_col"]
    sample = df.sample(min(len(df), n), random_state=42)
    cols = [
        id_col,
        "segment_name",
        "churn_prob",
        "predicted_clv",
        "priority_score",
        "pca_1",
        "pca_2",
        "recommended_offer",
        "offer_accept_prob",
    ]
    cols = [c for c in cols if c in sample.columns]
    out = sample[cols].rename(columns={id_col: "customer_id"})
    return out.to_dict(orient="records")
