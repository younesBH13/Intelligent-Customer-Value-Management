from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DATASETS, RANDOM_STATE


def run_nlp_dissatisfaction_analysis(dataset_key: str, n_topics: int = 5) -> Dict[str, Any]:
    """
    Optional NLP Extension:
    Extract recurring dissatisfaction themes from Churn Reason / complaint text when available.
    """
    if dataset_key != "ibm_telco":
        return {
            "available": False,
            "note": (
                "No complaint or churn-reason text column is present in UCI Iranian Churn. "
                "When Ooredoo provides authorized complaint/ticket text, this module runs TF-IDF + NMF topic extraction."
            ),
            "topics": [],
            "top_reasons": [],
        }

    spec = DATASETS[dataset_key]
    df_raw = pd.read_csv(spec.file_path)
    reasons = df_raw["Churn Reason"].dropna().astype(str).str.strip()
    reasons = reasons[reasons != ""]

    if len(reasons) < 10:
        return {"available": False, "note": "Insufficient text records.", "topics": [], "top_reasons": []}

    # Exact reason frequency summary
    top_reasons_df = (
        reasons.value_counts()
        .reset_index()
        .rename(columns={"Churn Reason": "reason", "count": "count"})
        .head(12)
    )
    top_reasons_df["share_pct"] = (top_reasons_df["count"] / len(reasons) * 100.0).round(2)

    # TF-IDF + NMF topic discovery
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=300)
    tfidf = vec.fit_transform(reasons)
    terms = vec.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, max_iter=400)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    dominant_topic = W.argmax(axis=1)
    topics: List[Dict[str, Any]] = []
    for t_idx in range(n_topics):
        top_term_idx = H[t_idx].argsort()[::-1][:5]
        keywords = [str(terms[i]) for i in top_term_idx]
        count = int((dominant_topic == t_idx).sum())
        topics.append(
            {
                "topic_id": t_idx + 1,
                "theme_keywords": ", ".join(keywords),
                "document_count": count,
                "share_pct": round(count / len(reasons) * 100.0, 2),
            }
        )

    topics.sort(key=lambda d: d["document_count"], reverse=True)

    return {
        "available": True,
        "note": (
            f"Analyzed {len(reasons):,} post-churn reason records from IBM Telco using TF-IDF + NMF "
            "(excluded from predictive churn training to prevent target leakage)."
        ),
        "total_text_records": int(len(reasons)),
        "topics": topics,
        "top_reasons": top_reasons_df.to_dict(orient="records"),
    }

