from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import OFFER_CATALOG, RANDOM_STATE


def compute_content_match_score(
    row: pd.Series,
    offer_id: str,
    dataset_key: str,
) -> float:
    """
    Content-based filtering score in [0, 1] measuring how well an offer
    matches a subscriber's consumption, contract, and service behaviour.
    """
    if dataset_key == "iranian":
        has_issue = 1.0 if (row.get("complains", 0) > 0 or row.get("call_failure", 0) >= 5) else 0.0
        high_usage = min(1.0, float(row.get("seconds_of_use", 0)) / 8000.0)
        high_charge = min(1.0, float(row.get("charge_amount", 0)) / 5.0)
        long_tenure = min(1.0, float(row.get("subscription_length", 0)) / 36.0)
        if offer_id == "VIP_CONCIERGE":
            return round(0.45 * high_charge + 0.35 * long_tenure + 0.20 * high_usage, 4)
        if offer_id == "SERVICE_RECOVERY":
            return round(0.80 * has_issue + 0.20 * (1.0 - high_charge), 4)
        if offer_id == "CONTRACT_LOCK":
            return round(0.50 * (1.0 - long_tenure) + 0.50 * high_charge, 4)
        if offer_id == "USAGE_BUNDLE":
            return round(0.70 * high_usage + 0.30 * (1.0 - has_issue), 4)
        return 0.35

    # ibm_telco
    is_mtm = 1.0 if row.get("is_long_term_contract", 0) == 0 else 0.0
    high_charge = min(1.0, float(row.get("monthly_charges", 20.0)) / 100.0)
    no_sec = 1.0 if row.get("has_security_bundle", 0) == 0 else 0.0
    has_stream = float(row.get("has_streaming", 0))

    if offer_id == "VIP_CONCIERGE":
        return round(0.50 * high_charge + 0.30 * (1.0 - is_mtm) + 0.20 * has_stream, 4)
    if offer_id == "CONTRACT_LOCK":
        return round(0.70 * is_mtm + 0.30 * high_charge, 4)
    if offer_id == "SERVICE_RECOVERY":
        return round(0.65 * no_sec + 0.35 * high_charge, 4)
    if offer_id == "USAGE_BUNDLE":
        return round(0.55 * has_stream + 0.45 * (1.0 - high_charge), 4)
    return 0.35


def generate_synthetic_campaign_data(
    df_clean: pd.DataFrame,
    dataset_key: str,
    id_col: str,
    churn_probs: np.ndarray,
    predicted_clv: np.ndarray,
) -> pd.DataFrame:
    """
    Source B — Synthetic Telecom Business Data Generator.
    Generates realistic historical campaign exposure, offer acceptance,
    and post-campaign retention/revenue outcomes for evaluating Module 3 & Module 4.
    NOTE: Explicitly labeled as synthetic simulation data, not observed Ooredoo responses.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df_clean)

    offer_ids = [o.offer_id for o in OFFER_CATALOG]
    offer_map = {o.offer_id: o for o in OFFER_CATALOG}
    channels = ["SMS", "Mobile App Push", "Outbound Call", "Email"]

    records: List[Dict[str, Any]] = []
    for idx in range(n):
        row = df_clean.iloc[idx]
        c_prob = float(churn_probs[idx])
        c_val = float(predicted_clv[idx])

        # Assign a historical campaign offer (slightly biased toward plausible offers)
        chosen_id = str(rng.choice(offer_ids))
        offer = offer_map[chosen_id]
        match_score = compute_content_match_score(row, chosen_id, dataset_key)
        channel = str(rng.choice(channels, p=[0.35, 0.35, 0.15, 0.15]))

        # Latent acceptance probability driven by content match, churn risk, and offer cost/generosity
        channel_boost = 0.06 if channel in ("Outbound Call", "Mobile App Push") else 0.0
        logit = -1.1 + 2.1 * match_score + 0.6 * c_prob + channel_boost - 0.012 * offer.cost
        p_accept = float(1.0 / (1.0 + np.exp(-logit)))
        accepted = int(rng.random() < p_accept)

        # Post-campaign retention & revenue outcome (simulated)
        if accepted == 1:
            retain_prob = max(0.05, 1.0 - c_prob * (1.0 - offer.expected_retention_lift))
        else:
            retain_prob = max(0.05, 1.0 - c_prob)
        retained = int(rng.random() < retain_prob)

        post_revenue = round(
            (c_val * 0.25 * (1.08 if accepted and chosen_id == "USAGE_BUNDLE" else 1.0))
            if retained
            else 0.0,
            2,
        )

        records.append(
            {
                "campaign_id": f"SYN-CMP-{(idx % 12) + 1:02d}",
                id_col: str(row[id_col]),
                "offer_id": chosen_id,
                "offer_name": offer.name,
                "exposure_channel": channel,
                "offer_cost": offer.cost,
                "content_match_score": match_score,
                "churn_prob_at_exposure": round(c_prob, 4),
                "predicted_clv_at_exposure": round(c_val, 2),
                "accepted": accepted,
                "retained": retained,
                "post_campaign_revenue": post_revenue,
                "data_source": "Source B (Synthetic Campaign Simulation)",
            }
        )

    return pd.DataFrame(records)

