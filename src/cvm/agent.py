from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR
from .digital_twin import simulate_digital_twin
from .explainability import explain_single_customer


def _extract_customer_id(query: str, df: pd.DataFrame, id_col: str) -> str | None:
    # Match IRN-0689 or IBM IDs like 5178-LMXOP
    irn_match = re.search(r"\bIRN-\d{4}\b", query, re.IGNORECASE)
    if irn_match:
        cid = irn_match.group(0).upper()
        if (df[id_col].astype(str) == cid).any():
            return cid

    ibm_match = re.search(r"\b\d{4}-[A-Z0-9]{5}\b", query, re.IGNORECASE)
    if ibm_match:
        cid = ibm_match.group(0).upper()
        if (df[id_col].astype(str) == cid).any():
            return cid
    return None


def ask_cvm_agent(dataset_key: str, query: str) -> Dict[str, Any]:
    """
    Deterministic, tool-grounded CVM AI Copilot.
    Answers portfolio, customer-explanation, segment, model, and Digital Twin
    what-if queries directly from trained models and scored artifacts.
    """
    out_dir = ARTIFACTS_DIR / dataset_key
    with open(out_dir / "report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    df = pd.read_csv(out_dir / "scored_customers.csv")
    id_col = report["dataset"]["id_col"]
    feature_cols = report["dataset"]["feature_cols"]
    q_lower = query.lower()

    cid = _extract_customer_id(query, df, id_col)

    # 1. Digital Twin What-If simulation intent
    if cid and any(w in q_lower for w in ["what if", "simulate", "twin", "switch", "resolve", "reduce", "if "]):
        overrides: Dict[str, Any] = {}
        if dataset_key == "iranian":
            if "complaint" in q_lower or "complain" in q_lower:
                overrides["complains"] = 0
            if "failure" in q_lower or "call_failure" in q_lower:
                overrides["call_failure"] = 0
            if "charge" in q_lower:
                m = re.search(r"charge[s]?\s*(?:=|to)?\s*(\d+(?:\.\d+)?)", q_lower)
                if m:
                    overrides["charge_amount"] = float(m.group(1))
        elif dataset_key == "ibm_telco":
            if "one year" in q_lower or "1-year" in q_lower or "1 year" in q_lower:
                overrides["contract_type"] = "One year"
            elif "two year" in q_lower or "2-year" in q_lower or "2 year" in q_lower:
                overrides["contract_type"] = "Two year"
            if "tech support" in q_lower or "security" in q_lower:
                overrides["tech_support_yes"] = 1
                overrides["online_security_yes"] = 1

        if not overrides:
            overrides = (
                {"complains": 0, "call_failure": 0}
                if dataset_key == "iranian"
                else {"contract_type": "One year", "tech_support_yes": 1}
            )

        sim = simulate_digital_twin(dataset_key, overrides=overrides, base_customer_id=cid)
        b = sim["baseline"]
        s = sim["simulated"]
        d = sim["deltas"]
        answer = (
            f"**Digital Twin Simulation for `{cid}`** (applied `{overrides}`):\n"
            f"- **Churn Risk**: `{b['churn_prob']:.1%}` → **`{s['churn_prob']:.1%}`** (`{d['delta_churn_prob']:+.1%}`)\n"
            f"- **Predicted Value**: `{b['predicted_clv']:,.1f}` → **`{s['predicted_clv']:,.1f}`** (`{d['delta_clv']:+,.1f}`)\n"
            f"- **Priority Score**: `{b['priority_score']:,.1f}` → **`{s['priority_score']:,.1f}`** (`{d['delta_priority']:+,.1f}`)\n"
            f"- **Updated Offer**: **{s['recommended_offer']}** (Sim. Net Benefit: `{s['sim_expected_net_benefit']:,.1f}`)"
        )
        return {"intent": "digital_twin_simulation", "answer": answer, "data": sim}

    # 2. Specific customer explanation & retention brief
    if cid:
        row_df = df[df[id_col].astype(str) == cid].iloc[[0]]
        r = row_df.iloc[0]
        churn_model = joblib.load(out_dir / "churn_model.joblib")
        shap_items = explain_single_customer(churn_model, row_df, feature_cols, top_k=5, background_df=df)
        drivers_txt = ", ".join(
            f"`{x['feature']}={x['feature_value']}` ({x['shap_contribution']:+.3f})"
            for x in shap_items[:3]
        )
        answer = (
            f"**Customer Retention Brief — `{cid}`**\n"
            f"- **Segment**: {r['segment_name']} | **Priority Rank**: #{int(r['priority_rank'])}\n"
            f"- **Churn Risk**: `{r['churn_prob']:.1%}` | **Predicted Value**: `{r['predicted_clv']:,.1f}` | **Priority**: `{r['priority_score']:,.1f}`\n"
            f"- **Top Churn Drivers (SHAP)**: {drivers_txt}\n"
            f"- **Recommended Action**: **{r['recommended_offer']}** ({r['recommended_action']})"
        )
        return {"intent": "customer_brief", "answer": answer, "data": {"customer_id": cid, "shap": shap_items}}

    # 3. Top priority customers
    if any(w in q_lower for w in ["top", "priority", "risk", "who", "target", "urgent"]):
        top5 = df.head(5)
        lines = [
            f"- **#{int(r['priority_rank'])} `{r[id_col]}`** ({r['segment_name']}): "
            f"P(Churn)=`{r['churn_prob']:.1%}`, Value=`{r['predicted_clv']:,.1f}`, "
            f"Priority=`{r['priority_score']:,.1f}` → *{r['recommended_offer']}*"
            for _, r in top5.iterrows()
        ]
        answer = f"**Top 5 Value-Aware Priority Customers ({report['dataset']['name']}):**\n" + "\n".join(lines)
        return {"intent": "top_priority", "answer": answer, "data": top5[[id_col, "churn_prob", "predicted_clv", "priority_score"]].to_dict(orient="records")}

    # 4. Segment summary
    if any(w in q_lower for w in ["segment", "cluster", "group"]):
        profs = report["stage2_segmentation"]["cluster_profiles"]
        lines = [
            f"- **{p['segment_name']}**: {p['customer_count']:,} customers ({p['share_pct']}%), "
            f"Avg Value=`{p['avg_value']:,.1f}`, Churn Rate=`{p['churn_rate']:.1%}`"
            for p in profs
        ]
        answer = f"**Customer Segments ({report['stage2_segmentation']['best_model']}):**\n" + "\n".join(lines)
        return {"intent": "segments", "answer": answer, "data": profs}

    # 5. Default: portfolio & model overview
    pol = report["stage5_recommendations"]["policy_comparison"]
    answer = (
        f"**CVM Portfolio Summary ({report['dataset']['name']})**\n"
        f"- **Customers**: `{len(df):,}` | **Observed Churn Rate**: `{report['stage1_data_quality']['churn_rate']:.1%}`\n"
        f"- **Best Churn Model**: `{report['stage4_churn_benchmark']['best_model']}` | "
        f"**Best CLV Model**: `{report['stage3_clv_benchmark']['best_model']}`\n"
        f"- **Simulated Value-Aware ROI (Top 30%)**: `{pol['value_aware_cvm_policy']['sim_roi_pct']}%` "
        f"vs. Generic Baseline `{pol['generic_churn_only_baseline']['sim_roi_pct']}%`\n\n"
        "Try asking: *'Show top priority customers'*, *'Compare segments'*, "
        f"or *'Simulate digital twin for {df.iloc[0][id_col]}'*."
    )
    return {"intent": "summary", "answer": answer, "data": pol}

