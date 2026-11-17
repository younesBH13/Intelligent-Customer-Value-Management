from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .data_prep import PreparedDataset


@dataclass
class SegmentationResult:
    best_model_name: str
    n_clusters: int
    metrics_table: List[Dict[str, Any]]
    cluster_profiles: List[Dict[str, Any]]
    labels: np.ndarray
    segment_names: List[str]
    pca_2d: np.ndarray
    scaler: StandardScaler
    model: Any


def _assign_business_labels(
    df_seg: pd.DataFrame,
    cluster_col: str,
    rfm_cols: List[str],
    value_col: str,
    churn_col: str,
) -> Dict[int, str]:
    """Map numeric cluster IDs to interpretable business segment labels post-hoc."""
    summary = (
        df_seg.groupby(cluster_col)
        .agg(
            mean_value=(value_col, "mean"),
            mean_tenure=(rfm_cols[0], "mean"),
            mean_freq=(rfm_cols[1], "mean"),
            churn_rate=(churn_col, "mean"),
            size=(value_col, "count"),
        )
        .reset_index()
    )

    overall_val = df_seg[value_col].median()
    overall_churn = df_seg[churn_col].mean()
    overall_tenure = df_seg[rfm_cols[0]].median()

    used_labels = set()
    mapping: Dict[int, str] = {}

    # Sort clusters by value descending so high-value clusters get priority naming
    summary = summary.sort_values("mean_value", ascending=False).reset_index(drop=True)

    for _, row in summary.iterrows():
        cid = int(row[cluster_col])
        high_val = row["mean_value"] >= overall_val
        high_risk = row["churn_rate"] >= max(overall_churn * 1.15, overall_churn + 0.03)
        high_tenure = row["mean_tenure"] >= overall_tenure

        if high_val and high_risk:
            candidate = "High-Value At Risk"
        elif high_val and not high_risk:
            candidate = "High-Value Loyal"
        elif not high_val and not high_tenure and not high_risk:
            candidate = "Growing"
        elif not high_val and high_risk:
            candidate = "Inactive / Declining"
        else:
            candidate = "Occasional"

        # Disambiguate if two clusters land on the same label
        final_label = candidate
        suffix = 2
        while final_label in used_labels:
            final_label = f"{candidate} ({suffix})"
            suffix += 1
        used_labels.add(final_label)
        mapping[cid] = final_label

    return mapping


def run_segmentation(
    prepared: PreparedDataset,
    n_clusters: int = 5,
) -> SegmentationResult:
    df = prepared.df_clean.copy()
    X_rfm = df[prepared.rfm_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rfm)

    candidates: Dict[str, Tuple[Any, np.ndarray]] = {}

    # 1. K-Means (baseline)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    candidates["KMeans (Baseline)"] = (km, km.fit_predict(X_scaled))

    # 2. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE, n_init=3)
    candidates["GaussianMixture"] = (gmm, gmm.fit_predict(X_scaled))

    # 3. Hierarchical Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    candidates["Agglomerative (Ward)"] = (agg, agg.fit_predict(X_scaled))

    # 4. DBSCAN (density-based / outlier detection)
    dbscan = DBSCAN(eps=0.9, min_samples=15)
    candidates["DBSCAN"] = (dbscan, dbscan.fit_predict(X_scaled))

    metrics_table: List[Dict[str, Any]] = []
    best_name = "KMeans (Baseline)"
    best_sil = -1.0

    for name, (model, labels) in candidates.items():
        unique_labels = set(labels)
        valid_mask = labels != -1
        n_valid_clusters = len(set(labels[valid_mask])) if valid_mask.any() else 0
        noise_pct = float((labels == -1).mean() * 100.0)

        if n_valid_clusters >= 2 and valid_mask.sum() > n_valid_clusters:
            sil = float(silhouette_score(X_scaled[valid_mask], labels[valid_mask]))
            dbi = float(davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask]))
            chi = float(calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask]))
        else:
            sil, dbi, chi = float("nan"), float("nan"), float("nan")

        metrics_table.append(
            {
                "algorithm": name,
                "n_clusters": int(n_valid_clusters),
                "noise_pct": round(noise_pct, 2),
                "silhouette": round(sil, 4) if not np.isnan(sil) else None,
                "davies_bouldin": round(dbi, 4) if not np.isnan(dbi) else None,
                "calinski_harabasz": round(chi, 2) if not np.isnan(chi) else None,
            }
        )

        # Select best among fixed-k partitioning models (KMeans, GMM, Agglomerative)
        if name != "DBSCAN" and not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_name = name

    best_model, best_labels = candidates[best_name]
    df["_cluster_id"] = best_labels

    label_map = _assign_business_labels(
        df_seg=df,
        cluster_col="_cluster_id",
        rfm_cols=prepared.rfm_cols,
        value_col=prepared.value_col,
        churn_col=prepared.churn_col,
    )
    segment_names = [label_map[int(c)] for c in best_labels]
    df["_segment_name"] = segment_names

    # Build rich cluster profiles
    agg_dict: Dict[str, Any] = {
        prepared.id_col: "count",
        prepared.value_col: "mean",
        prepared.churn_col: "mean",
    }
    for col in prepared.rfm_cols:
        if col not in (prepared.value_col,):
            agg_dict[col] = "mean"

    grouped = df.groupby(["_cluster_id", "_segment_name"]).agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={prepared.id_col: "customer_count"})
    grouped["share_pct"] = (grouped["customer_count"] / len(df) * 100.0).round(2)

    cluster_profiles: List[Dict[str, Any]] = []
    for _, r in grouped.sort_values(prepared.value_col, ascending=False).iterrows():
        prof = {
            "cluster_id": int(r["_cluster_id"]),
            "segment_name": str(r["_segment_name"]),
            "customer_count": int(r["customer_count"]),
            "share_pct": float(r["share_pct"]),
            "avg_value": round(float(r[prepared.value_col]), 2),
            "churn_rate": round(float(r[prepared.churn_col]), 4),
        }
        for col in prepared.rfm_cols:
            if col != prepared.value_col:
                prof[f"avg_{col}"] = round(float(r[col]), 2)
        cluster_profiles.append(prof)

    # 2D PCA projection for interactive visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_2d = pca.fit_transform(X_scaled)

    return SegmentationResult(
        best_model_name=best_name,
        n_clusters=n_clusters,
        metrics_table=metrics_table,
        cluster_profiles=cluster_profiles,
        labels=best_labels,
        segment_names=segment_names,
        pca_2d=pca_2d,
        scaler=scaler,
        model=best_model,
    )

