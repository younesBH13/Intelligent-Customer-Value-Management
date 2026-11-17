from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import OFFER_CATALOG, OfferSpec, RANDOM_STATE
from .synthetic_campaign import compute_content_match_score, generate_synthetic_campaign_data


@dataclass
class RecommendationSummary:
    scored_df: pd.DataFrame
    synthetic_campaigns_df: pd.DataFrame
    offer_acceptance_metrics: Dict[str, Any]
    policy_comparison: Dict[str, Any]
    offer_distribution: List[Dict[str, Any]]


def _is_eligible_for_offer(
    offer: OfferSpec,
    churn_prob: float,
    value_quantile: float,
    segment_name: str,
    row: pd.Series,
    dataset_key: str,
) -> bool:
    if churn_prob < offer.min_churn_prob:
        return False
    if value_quantile < offer.min_value_quantile:
        return False

    base_seg = segment_name.split(" (")[0]
    if base_seg not in offer.target_segments:
        return False

    if offer.offer_id == "SERVICE_RECOVERY":
        if dataset_key == "iranian":
            has_issue = (row.get("complains", 0) > 0) or (row.get("call_failure", 0) >= 5)
            if not has_issue:
                return False
        elif dataset_key == "ibm_telco":
            if row.get("has_security_bundle", 0) == 1 and churn_prob < 0.55:
                return False

    if offer.offer_id == "CONTRACT_LOCK" and dataset_key == "ibm_telco":
        if row.get("is_long_term_contract", 0) == 1:
            return False

    return True


def _train_offer_acceptance_model(
    campaigns_df: pd.DataFrame,
) -> Tuple[LGBMClassifier, List[str], Dict[str, Any]]:
    """
    Module 3: Train Offer Acceptance Predictor on Source B synthetic campaign logs.
    """
    offer_ids = [o.offer_id for o in OFFER_CATALOG]
    df_feat = campaigns_df[
        ["content_match_score", "churn_prob_at_exposure", "predicted_clv_at_exposure", "offer_cost"]
    ].copy()
    for oid in offer_ids:
        df_feat[f"offer_{oid}"] = (campaigns_df["offer_id"] == oid).astype(int)

    feature_names = list(df_feat.columns)
    y = campaigns_df["accepted"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df_feat, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    model = LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "model": "LightGBM Offer Acceptance Predictor (Source B Synthetic)",
        "is_synthetic_source_b": True,
        "observed_acceptance_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "note": (
            "Trained on Source B synthetic campaign exposure logs to demonstrate Module 3 "
            "(Offer Acceptance Prediction). Cannot be claimed as real-world campaign effectiveness."
        ),
    }
    return model, feature_names, metrics


def run_recommendations(
    df_clean: pd.DataFrame,
    dataset_key: str,
    id_col: str,
    churn_probs: np.ndarray,
    predicted_clv: np.ndarray,
    segment_names: List[str],
    cluster_ids: np.ndarray,
    pca_2d: np.ndarray,
) -> RecommendationSummary:
    df = df_clean.copy()
    df["churn_prob"] = np.round(churn_probs, 4)
    df["predicted_clv"] = np.round(predicted_clv, 2)
    df["priority_score"] = np.round(df["churn_prob"] * df["predicted_clv"], 2)
    df["segment_name"] = segment_names
    df["cluster_id"] = cluster_ids
    df["pca_1"] = np.round(pca_2d[:, 0], 4)
    df["pca_2"] = np.round(pca_2d[:, 1], 4)

    # Generate Source B synthetic campaign data & train Module 3 Offer Acceptance model
    campaigns_df = generate_synthetic_campaign_data(
        df_clean=df,
        dataset_key=dataset_key,
        id_col=id_col,
        churn_probs=churn_probs,
        predicted_clv=predicted_clv,
    )
    accept_model, accept_cols, accept_metrics = _train_offer_acceptance_model(campaigns_df)

    # Batch-score P(Accept) for every (customer, offer) pair for fast Module 4 NBA selection
    offer_ids = [o.offer_id for o in OFFER_CATALOG]
    pair_accept_probs: Dict[str, np.ndarray] = {}
    pair_match_scores: Dict[str, np.ndarray] = {}

    for offer in OFFER_CATALOG:
        m_scores = np.array(
            [
                compute_content_match_score(df.iloc[i], offer.offer_id, dataset_key)
                for i in range(len(df))
            ]
        )
        pair_match_scores[offer.offer_id] = m_scores

        X_pair = pd.DataFrame(
            {
                "content_match_score": m_scores,
                "churn_prob_at_exposure": df["churn_prob"].values,
                "predicted_clv_at_exposure": df["predicted_clv"].values,
                "offer_cost": np.full(len(df), offer.cost),
            }
        )
        for oid in offer_ids:
            X_pair[f"offer_{oid}"] = int(oid == offer.offer_id)
        pair_accept_probs[offer.offer_id] = accept_model.predict_proba(X_pair[accept_cols])[:, 1]

    val_q = df["predicted_clv"].rank(pct=True)
    df["value_tier"] = pd.cut(
        val_q,
        bins=[0.0, 0.33, 0.67, 1.0],
        labels=["Low Value", "Medium Value", "High Value"],
        include_lowest=True,
    ).astype(str)

    df["risk_tier"] = pd.cut(
        df["churn_prob"],
        bins=[0.0, 0.25, 0.55, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    ).astype(str)

    rec_offer_ids: List[str] = []
    rec_offer_names: List[str] = []
    rec_action_types: List[str] = []
    rec_costs: List[float] = []
    rec_accept_probs: List[float] = []
    rec_match_scores: List[float] = []
    rec_expected_saved_val: List[float] = []
    rec_net_benefit: List[float] = []
    rec_rationales: List[str] = []

    for pos, (idx, row) in enumerate(df.iterrows()):
        c_prob = float(row["churn_prob"])
        c_val = float(row["predicted_clv"])
        v_q = float(val_q.loc[idx])
        seg = str(row["segment_name"])

        best_offer = OFFER_CATALOG[-1]
        best_utility = -1e9
        best_p_accept = float(pair_accept_probs[best_offer.offer_id][pos])
        best_match = float(pair_match_scores[best_offer.offer_id][pos])

        # Module 4: Utility-Based Next Best Action Decision Engine
        for offer in OFFER_CATALOG:
            if _is_eligible_for_offer(offer, c_prob, v_q, seg, row, dataset_key):
                p_acc = float(pair_accept_probs[offer.offer_id][pos])
                m_sc = float(pair_match_scores[offer.offer_id][pos])
                expected_saved = c_prob * c_val * p_acc * offer.expected_retention_lift
                utility = expected_saved - offer.cost
                if utility > best_utility:
                    best_utility = utility
                    best_offer = offer
                    best_p_accept = p_acc
                    best_match = m_sc

        expected_saved = round(
            c_prob * c_val * best_p_accept * best_offer.expected_retention_lift, 2
        )
        net_benefit = round(expected_saved - best_offer.cost, 2)

        rationale = (
            f"Segment '{seg}' | P(Churn)={c_prob:.1%}, Value={c_val:.1f}, "
            f"P(Accept)={best_p_accept:.1%}, Match={best_match:.2f}. "
            f"NBA Selected: {best_offer.name} (Cost={best_offer.cost:.0f}, Expected Utility={net_benefit:.1f})."
        )

        rec_offer_ids.append(best_offer.offer_id)
        rec_offer_names.append(best_offer.name)
        rec_action_types.append(best_offer.action_type)
        rec_costs.append(best_offer.cost)
        rec_accept_probs.append(round(best_p_accept, 4))
        rec_match_scores.append(round(best_match, 4))
        rec_expected_saved_val.append(expected_saved)
        rec_net_benefit.append(net_benefit)
        rec_rationales.append(rationale)

    df["recommended_offer_id"] = rec_offer_ids
    df["recommended_offer"] = rec_offer_names
    df["recommended_action"] = rec_action_types
    df["offer_cost"] = rec_costs
    df["offer_accept_prob"] = rec_accept_probs
    df["content_match_score"] = rec_match_scores
    df["sim_expected_saved_value"] = rec_expected_saved_val
    df["sim_expected_net_benefit"] = rec_net_benefit
    df["recommendation_rationale"] = rec_rationales

    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)

    top_k = max(1, int(len(df) * 0.30))
    generic_top = df.sort_values("churn_prob", ascending=False).head(top_k)
    generic_cost = float(top_k * 15.0)
    generic_saved = float((generic_top["churn_prob"] * generic_top["predicted_clv"] * 0.20 * 0.45).sum())
    generic_net = generic_saved - generic_cost

    cvm_top = df.head(top_k)
    cvm_cost = float(cvm_top["offer_cost"].sum())
    cvm_saved = float(cvm_top["sim_expected_saved_value"].sum())
    cvm_net = cvm_saved - cvm_cost

    policy_comparison = {
        "note": (
            "Simulated policy comparison (top 30% targeted) combining Source A models with "
            "Source B learned Offer Acceptance probabilities. Illustrative simulation only."
        ),
        "targeted_customers": int(top_k),
        "generic_churn_only_baseline": {
            "total_cost": round(generic_cost, 2),
            "sim_saved_value": round(generic_saved, 2),
            "sim_net_value": round(generic_net, 2),
            "sim_roi_pct": round((generic_net / generic_cost * 100.0) if generic_cost > 0 else 0.0, 1),
        },
        "value_aware_cvm_policy": {
            "total_cost": round(cvm_cost, 2),
            "sim_saved_value": round(cvm_saved, 2),
            "sim_net_value": round(cvm_net, 2),
            "sim_roi_pct": round((cvm_net / cvm_cost * 100.0) if cvm_cost > 0 else 0.0, 1),
        },
        "total_portfolio_value_at_risk": round(float(df["priority_score"].sum()), 2),
    }

    offer_dist = (
        df.groupby(["recommended_offer_id", "recommended_offer"])
        .agg(
            customer_count=(id_col, "count"),
            avg_churn_prob=("churn_prob", "mean"),
            avg_accept_prob=("offer_accept_prob", "mean"),
            avg_predicted_clv=("predicted_clv", "mean"),
            total_cost=("offer_cost", "sum"),
        )
        .reset_index()
    )
    offer_dist["share_pct"] = (offer_dist["customer_count"] / len(df) * 100.0).round(2)
    offer_distribution = offer_dist.sort_values("customer_count", ascending=False).to_dict(orient="records")

    return RecommendationSummary(
        scored_df=df,
        synthetic_campaigns_df=campaigns_df,
        offer_acceptance_metrics=accept_metrics,
        policy_comparison=policy_comparison,
        offer_distribution=offer_distribution,
    )
