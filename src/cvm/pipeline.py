from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import joblib
import mlflow

from .churn_models import run_churn_benchmark
from .clv_models import run_clv_benchmark
from .config import ARTIFACTS_DIR, DATASETS, ROOT_DIR
from .data_prep import load_and_prepare
from .database import persist_pipeline_to_db
from .explainability import compute_global_importance
from .nlp_insights import run_nlp_dissatisfaction_analysis
from .recommender import run_recommendations
from .segmentation import run_segmentation
from .uplift import run_simulated_uplift


def run_pipeline(dataset_key: str, n_clusters: int = 5) -> Dict[str, Any]:
    prepared = load_and_prepare(dataset_key)

    # Stage 2: Customer Segmentation
    seg_res = run_segmentation(prepared, n_clusters=n_clusters)

    # Stage 3: CLV / Value Prediction
    clv_res = run_clv_benchmark(prepared)

    # Stage 4: Churn Prediction (with CatBoost, 5-fold CV, Calibration)
    churn_res = run_churn_benchmark(prepared)

    # Stage 5: Value-Aware Prioritization & Next Best Offer
    rec_res = run_recommendations(
        df_clean=prepared.df_clean,
        dataset_key=dataset_key,
        id_col=prepared.id_col,
        churn_probs=churn_res.predicted_probs,
        predicted_clv=clv_res.predicted_values,
        segment_names=seg_res.segment_names,
        cluster_ids=seg_res.labels,
        pca_2d=seg_res.pca_2d,
    )

    # Stage 5b: Semi-Synthetic Uplift Modeling (S/T/X-Learner)
    uplift_res = run_simulated_uplift(prepared, churn_res.predicted_probs)

    # Stage 6: Explainable AI (Global SHAP)
    X_features = prepared.df_clean[prepared.feature_cols]
    churn_shap_global = compute_global_importance(
        churn_res.best_model, X_features, prepared.feature_cols
    )
    clv_shap_global = compute_global_importance(
        clv_res.best_model, X_features, prepared.feature_cols
    )

    # Optional NLP Extension (dissatisfaction themes)
    nlp_res = run_nlp_dissatisfaction_analysis(dataset_key)

    out_dir = ARTIFACTS_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / "scored_customers.csv"
    rec_res.scored_df.to_csv(scored_path, index=False)
    rec_res.synthetic_campaigns_df.to_csv(out_dir / "synthetic_campaigns.csv", index=False)

    joblib.dump(churn_res.best_model, out_dir / "churn_model.joblib")
    joblib.dump(clv_res.best_model, out_dir / "clv_model.joblib")

    report: Dict[str, Any] = {
        "dataset": {
            "key": prepared.key,
            "name": prepared.spec.name,
            "description": prepared.spec.description,
            "id_col": prepared.id_col,
            "churn_col": prepared.churn_col,
            "value_col": prepared.value_col,
            "value_label": prepared.spec.value_label,
            "feature_cols": prepared.feature_cols,
            "rfm_cols": prepared.rfm_cols,
        },
        "stage1_data_quality": prepared.quality_report,
        "stage2_segmentation": {
            "best_model": seg_res.best_model_name,
            "n_clusters": seg_res.n_clusters,
            "metrics_table": seg_res.metrics_table,
            "cluster_profiles": seg_res.cluster_profiles,
        },
        "stage3_clv_benchmark": {
            "best_model": clv_res.best_model_name,
            "metrics_table": clv_res.metrics_table,
        },
        "stage4_churn_benchmark": {
            "best_model": churn_res.best_model_name,
            "metrics_table": churn_res.metrics_table,
            "confusion_matrix": churn_res.confusion_matrix,
            "roc_curves": churn_res.roc_curves,
            "pr_curves": churn_res.pr_curves,
            "calibration_curves": churn_res.calibration_curves,
        },
        "stage5_recommendations": {
            "offer_acceptance_model": rec_res.offer_acceptance_metrics,
            "policy_comparison": rec_res.policy_comparison,
            "offer_distribution": rec_res.offer_distribution,
        },
        "stage5b_uplift_simulation": uplift_res,
        "stage6_explainability": {
            "note": (
                "SHAP values explain model behavior (feature contributions to predicted risk/value), "
                "not causal reasons why a customer churns or whether an offer will causally retain them."
            ),
            "churn_global_importance": churn_shap_global,
            "clv_global_importance": clv_shap_global,
        },
        "optional_nlp_insights": nlp_res,
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # MLflow Experiment Tracking (SQLite DB backend per MLflow 3.x)
    mlflow_db_uri = f"sqlite:///{(ARTIFACTS_DIR / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(mlflow_db_uri)
    mlflow.set_experiment("Intelligent_CVM")

    mlflow_run_id = None
    with mlflow.start_run(run_name=f"cvm_{dataset_key}") as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_params(
            {
                "dataset_key": dataset_key,
                "n_rows": prepared.quality_report["clean_rows"],
                "n_clusters": n_clusters,
                "best_segmentation": seg_res.best_model_name,
                "best_clv_model": clv_res.best_model_name,
                "best_churn_model": churn_res.best_model_name,
                "best_uplift_learner": uplift_res["best_learner"],
            }
        )
        best_churn_row = max(churn_res.metrics_table, key=lambda x: x["pr_auc"])
        best_clv_row = min(clv_res.metrics_table, key=lambda x: x["rmse"])
        best_seg_row = next(
            m for m in seg_res.metrics_table if m["algorithm"] == seg_res.best_model_name
        )
        mlflow.log_metrics(
            {
                "churn_pr_auc": best_churn_row["pr_auc"],
                "churn_roc_auc": best_churn_row["roc_auc"],
                "churn_brier": best_churn_row["brier_score"],
                "churn_ece": best_churn_row["ece"],
                "clv_rmse": best_clv_row["rmse"],
                "clv_r2": best_clv_row["r2"],
                "seg_silhouette": best_seg_row["silhouette"] or 0.0,
                "offer_accept_roc_auc": rec_res.offer_acceptance_metrics["roc_auc"],
                "sim_cvm_roi_pct": rec_res.policy_comparison["value_aware_cvm_policy"]["sim_roi_pct"],
            }
        )
        mlflow.log_artifact(str(report_path))

    # Database Persistence (PostgreSQL via CVM_DATABASE_URL or local SQLite)
    db_info = persist_pipeline_to_db(
        dataset_key=dataset_key,
        scored_df=rec_res.scored_df,
        report=report,
        mlflow_run_id=mlflow_run_id,
    )
    report["ml_engineering"] = {
        "mlflow_tracking_uri": mlflow_db_uri,
        "mlflow_run_id": mlflow_run_id,
        "database": db_info,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Intelligent CVM pipeline.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to run (iranian, ibm_telco, or all).",
    )
    parser.add_argument("--clusters", type=int, default=5, help="Number of customer segments.")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for k in keys:
        print(f"Running CVM pipeline for '{k}'...")
        rep = run_pipeline(k, n_clusters=args.clusters)
        print(
            f"  Done '{k}': rows={rep['stage1_data_quality']['clean_rows']}, "
            f"churn_best={rep['stage4_churn_benchmark']['best_model']}, "
            f"clv_best={rep['stage3_clv_benchmark']['best_model']}, "
            f"mlflow_run={rep['ml_engineering']['mlflow_run_id']}, "
            f"db={rep['ml_engineering']['database']['backend']}"
        )


if __name__ == "__main__":
    main()
