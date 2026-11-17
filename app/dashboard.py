from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cvm.agent import ask_cvm_agent  # noqa: E402
from cvm.config import ARTIFACTS_DIR, DATASETS  # noqa: E402
from cvm.digital_twin import simulate_digital_twin  # noqa: E402
from cvm.explainability import explain_single_customer  # noqa: E402

st.set_page_config(
    page_title="Intelligent CVM Decision-Support System",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_report(dataset_key: str) -> Dict[str, Any]:
    path = ARTIFACTS_DIR / dataset_key / "report.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_scored_df(dataset_key: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / dataset_key / "scored_customers.csv")


@st.cache_resource
def load_models(dataset_key: str):
    churn_m = joblib.load(ARTIFACTS_DIR / dataset_key / "churn_model.joblib")
    clv_m = joblib.load(ARTIFACTS_DIR / dataset_key / "clv_model.joblib")
    return churn_m, clv_m


def main() -> None:
    st.title("Intelligent Customer Value Management (CVM) — Decision-Support System")
    st.caption(
        "Value-aware, explainable CVM prototype: Segmentation, CLV Estimation, Churn Risk, "
        "Next Best Offer Prioritization & SHAP Explanations (Ooredoo PFE Architecture)"
    )

    available_keys = [k for k in DATASETS if (ARTIFACTS_DIR / k / "report.json").exists()]
    if not available_keys:
        st.error("No pipeline artifacts found. Run `python -m cvm.pipeline --dataset all` first.")
        return

    with st.sidebar:
        st.header("Configuration")
        dataset_key = st.selectbox(
            "Select Dataset",
            options=available_keys,
            format_func=lambda k: DATASETS[k].name,
        )
        report = load_report(dataset_key)
        df = load_scored_df(dataset_key)

        st.markdown("---")
        st.subheader("Dataset Scope")
        st.write(report["dataset"]["description"])
        st.info(
            f"**Customers:** {report['stage1_data_quality']['clean_rows']:,}\n\n"
            f"**Churn Rate:** {report['stage1_data_quality']['churn_rate']:.1%}\n\n"
            f"**Value Metric:** {report['dataset']['value_label']}"
        )

    tabs = st.tabs(
        [
            "1. Executive Overview",
            "2. Customer Segmentation",
            "3. Predictive Benchmark (CLV & Churn)",
            "4. Priority Queue & Next Best Offer",
            "5. Customer XAI Inspector (SHAP)",
            "6. Dual-Dataset Domain Shift",
            "7. Uplift Simulation & NLP",
            "8. Digital Twin & AI Agent",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: Executive Overview
    # ------------------------------------------------------------------
    with tabs[0]:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", f"{len(df):,}")
        k2.metric("Observed Churn Rate", f"{df[report['dataset']['churn_col']].mean():.1%}")
        k3.metric("Avg Predicted Value", f"{df['predicted_clv'].mean():,.1f}")
        k4.metric(
            "Total Value at Risk (Σ Priority)",
            f"{df['priority_score'].sum():,.0f}",
        )

        st.markdown("### Risk vs. Customer Value Matrix")
        sample_df = df.sample(min(len(df), 1500), random_state=42)
        fig_rv = px.scatter(
            sample_df,
            x="predicted_clv",
            y="churn_prob",
            color="segment_name",
            size="priority_score",
            hover_data=[report["dataset"]["id_col"], "recommended_offer", "priority_score"],
            labels={
                "predicted_clv": f"Predicted {report['dataset']['value_label']}",
                "churn_prob": "Predicted Churn Probability",
                "segment_name": "Segment",
            },
            title="Customer Portfolio: Predicted Value vs. Churn Risk (sized by Priority Score)",
        )
        st.plotly_chart(fig_rv, use_container_width=True)

        st.markdown("### Simulated Policy Comparison (Top 30% Targeted)")
        pol = report["stage5_recommendations"]["policy_comparison"]
        st.warning(pol["note"])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Generic Churn-Only Baseline")
            st.json(pol["generic_churn_only_baseline"])
        with c2:
            st.subheader("Value-Aware CVM Policy")
            st.json(pol["value_aware_cvm_policy"])

    # ------------------------------------------------------------------
    # TAB 2: Customer Segmentation
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader(
            f"Stage 2 — Clustering Benchmark (Selected: {report['stage2_segmentation']['best_model']})"
        )
        seg_metrics = pd.DataFrame(report["stage2_segmentation"]["metrics_table"])
        st.dataframe(seg_metrics, use_container_width=True)

        profiles_df = pd.DataFrame(report["stage2_segmentation"]["cluster_profiles"])
        col_a, col_b = st.columns([1.2, 1.0])
        with col_a:
            fig_pca = px.scatter(
                df.sample(min(len(df), 2000), random_state=42),
                x="pca_1",
                y="pca_2",
                color="segment_name",
                hover_data=[report["dataset"]["id_col"], "predicted_clv", "churn_prob"],
                title="2D PCA Projection of Customer Segments",
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        with col_b:
            fig_bar = px.bar(
                profiles_df,
                x="segment_name",
                y="share_pct",
                color="churn_rate",
                text="customer_count",
                color_continuous_scale="RdYlGn_r",
                title="Segment Share (%) Colored by Churn Rate",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Segment Profiles")
        st.dataframe(profiles_df, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: Predictive Benchmark (CLV & Churn)
    # ------------------------------------------------------------------
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(
                f"Stage 3 — CLV / Value Prediction (Best: {report['stage3_clv_benchmark']['best_model']})"
            )
            clv_df = pd.DataFrame(report["stage3_clv_benchmark"]["metrics_table"])
            st.dataframe(clv_df, use_container_width=True)

            clv_imp = pd.DataFrame(report["stage6_explainability"]["clv_global_importance"][:10])
            fig_clv_imp = px.bar(
                clv_imp[::-1],
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Top 10 Global SHAP Features (CLV Model)",
            )
            st.plotly_chart(fig_clv_imp, use_container_width=True)

        with col2:
            st.subheader(
                f"Stage 4 — Churn Prediction (Best: {report['stage4_churn_benchmark']['best_model']})"
            )
            churn_df = pd.DataFrame(report["stage4_churn_benchmark"]["metrics_table"])
            st.dataframe(churn_df, use_container_width=True)

            churn_imp = pd.DataFrame(report["stage6_explainability"]["churn_global_importance"][:10])
            fig_churn_imp = px.bar(
                churn_imp[::-1],
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Top 10 Global SHAP Features (Churn Model)",
            )
            st.plotly_chart(fig_churn_imp, use_container_width=True)

        # ROC, PR, and Calibration curves
        c_roc, c_pr, c_cal = st.columns(3)
        with c_roc:
            fig_roc = go.Figure()
            for m_name, pts in report["stage4_churn_benchmark"]["roc_curves"].items():
                fig_roc.add_trace(go.Scatter(x=pts["fpr"], y=pts["tpr"], mode="lines", name=m_name))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
            fig_roc.update_layout(title="ROC Curves (Churn)", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_pr:
            fig_pr = go.Figure()
            for m_name, pts in report["stage4_churn_benchmark"]["pr_curves"].items():
                fig_pr.add_trace(go.Scatter(x=pts["recall"], y=pts["precision"], mode="lines", name=m_name))
            fig_pr.update_layout(title="Precision-Recall Curves (Churn)", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)

        with c_cal:
            cal_curves = report["stage4_churn_benchmark"].get("calibration_curves", {})
            fig_cal = go.Figure()
            for m_name, pts in cal_curves.items():
                fig_cal.add_trace(go.Scatter(x=pts["prob_pred"], y=pts["prob_true"], mode="lines+markers", name=m_name))
            fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
            fig_cal.update_layout(title="Calibration Curves", xaxis_title="Mean Predicted Prob", yaxis_title="Observed Fraction")
            st.plotly_chart(fig_cal, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4: Priority Queue & Next Best Offer
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Module 3 & 4 — Offer Acceptance Prediction & Utility-Based Next Best Action")
        accept_info = report["stage5_recommendations"].get("offer_acceptance_model", {})
        if accept_info:
            st.info(
                f"**{accept_info.get('model')}** | "
                f"ROC-AUC: `{accept_info.get('roc_auc')}` | "
                f"PR-AUC: `{accept_info.get('pr_auc')}` | "
                f"F1: `{accept_info.get('f1')}` | "
                f"Observed Synthetic Acceptance Rate: `{accept_info.get('observed_acceptance_rate', 0):.1%}`\n\n"
                f"*{accept_info.get('note', '')}*"
            )

        f1, f2, f3 = st.columns(3)
        seg_filter = f1.multiselect("Segment", sorted(df["segment_name"].unique()))
        risk_filter = f2.multiselect("Risk Tier", ["High Risk", "Medium Risk", "Low Risk"])
        val_filter = f3.multiselect("Value Tier", ["High Value", "Medium Value", "Low Value"])

        filtered = df.copy()
        if seg_filter:
            filtered = filtered[filtered["segment_name"].isin(seg_filter)]
        if risk_filter:
            filtered = filtered[filtered["risk_tier"].isin(risk_filter)]
        if val_filter:
            filtered = filtered[filtered["value_tier"].isin(val_filter)]

        id_col = report["dataset"]["id_col"]
        display_cols = [
            "priority_rank",
            id_col,
            "segment_name",
            "churn_prob",
            "predicted_clv",
            "priority_score",
            "recommended_offer",
            "offer_accept_prob",
            "content_match_score",
            "offer_cost",
            "sim_expected_net_benefit",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(200), use_container_width=True)

        # Aggregate Export (No PII)
        agg_csv = (
            filtered.groupby(["segment_name", "recommended_offer"])
            .agg(
                customers=(id_col, "count"),
                avg_churn_prob=("churn_prob", "mean"),
                avg_predicted_clv=("predicted_clv", "mean"),
                total_offer_cost=("offer_cost", "sum"),
                total_sim_net_benefit=("sim_expected_net_benefit", "sum"),
            )
            .reset_index()
            .to_csv(index=False)
        )
        st.download_button(
            label="Export Aggregate Report (CSV, No PII)",
            data=agg_csv,
            file_name=f"{dataset_key}_aggregate_cvm_report.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------------
    # TAB 5: Customer XAI Inspector (SHAP)
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Individual Customer Explanation & Offer Rationale")
        st.caption(report["stage6_explainability"]["note"])

        id_col = report["dataset"]["id_col"]
        top_ids = df[id_col].astype(str).head(100).tolist()
        selected_id = st.selectbox("Select Customer ID (Top 100 by Priority)", options=top_ids)

        row_df = df[df[id_col].astype(str) == selected_id].iloc[[0]]
        r = row_df.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Segment", str(r["segment_name"]))
        m2.metric("Churn Risk", f"{r['churn_prob']:.1%}")
        m3.metric("Predicted Value", f"{r['predicted_clv']:,.1f}")
        m4.metric("Priority Score", f"{r['priority_score']:,.1f}")

        st.success(f"**Recommended Offer:** {r['recommended_offer']} ({r['recommended_action']})\n\n{r['recommendation_rationale']}")

        churn_model, clv_model = load_models(dataset_key)
        feature_cols = report["dataset"]["feature_cols"]

        c_left, c_right = st.columns(2)
        with c_left:
            churn_shap = pd.DataFrame(
                explain_single_customer(churn_model, row_df, feature_cols, top_k=8, background_df=df)
            )
            fig_cs = px.bar(
                churn_shap[::-1],
                x="shap_contribution",
                y="feature",
                orientation="h",
                color=churn_shap[::-1]["shap_contribution"] > 0,
                color_discrete_map={True: "#d62728", False: "#2ca02c"},
                title="Local SHAP — Churn Risk Drivers",
            )
            fig_cs.update_layout(showlegend=False)
            st.plotly_chart(fig_cs, use_container_width=True)

        with c_right:
            clv_shap = pd.DataFrame(
                explain_single_customer(clv_model, row_df, feature_cols, top_k=8, background_df=df)
            )
            fig_vs = px.bar(
                clv_shap[::-1],
                x="shap_contribution",
                y="feature",
                orientation="h",
                color=clv_shap[::-1]["shap_contribution"] > 0,
                color_discrete_map={True: "#1f77b4", False: "#ff7f0e"},
                title="Local SHAP — CLV / Value Drivers",
            )
            fig_vs.update_layout(showlegend=False)
            st.plotly_chart(fig_vs, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 6: Dual-Dataset Domain Shift
    # ------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Dual-Dataset Evaluation & Domain Shift Comparison")
        st.write(
            "Per the PFE methodology, datasets are **evaluated separately** and never merged naively, "
            "because customer populations, feature definitions, observation windows, and value targets differ."
        )
        comp_rows = []
        for k in available_keys:
            rep = load_report(k)
            best_churn = max(rep["stage4_churn_benchmark"]["metrics_table"], key=lambda x: x["pr_auc"])
            best_clv = min(rep["stage3_clv_benchmark"]["metrics_table"], key=lambda x: x["rmse"])
            comp_rows.append(
                {
                    "Dataset": rep["dataset"]["name"],
                    "Rows": rep["stage1_data_quality"]["clean_rows"],
                    "Churn Rate": f"{rep['stage1_data_quality']['churn_rate']:.1%}",
                    "Value Target": rep["dataset"]["value_label"],
                    "Best Segmentation": rep["stage2_segmentation"]["best_model"],
                    "Best Churn Model": f"{best_churn['model']} (PR-AUC={best_churn['pr_auc']}, ROC-AUC={best_churn['roc_auc']})",
                    "Best CLV Model": f"{best_clv['model']} (R²={best_clv['r2']}, RMSE={best_clv['rmse']})",
                }
            )
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 7: Uplift Simulation, NLP Extension & ML Engineering
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Semi-Synthetic Uplift Modeling (S / T / X-Learner)")
        uplift = report.get("stage5b_uplift_simulation", {})
        if uplift:
            st.warning(uplift.get("note", ""))
            u_col1, u_col2 = st.columns([1.0, 1.2])
            with u_col1:
                st.dataframe(pd.DataFrame(uplift["metrics_table"]), use_container_width=True)
            with u_col2:
                fig_up = go.Figure()
                for l_name, pts in uplift["uplift_curves"].items():
                    fig_up.add_trace(
                        go.Scatter(
                            x=pts["fractions"],
                            y=pts["cumulative_uplift"],
                            mode="lines+markers",
                            name=l_name,
                        )
                    )
                fig_up.update_layout(
                    title="Cumulative Uplift Curves (Simulated Retention Lift)",
                    xaxis_title="Targeted Population Fraction",
                    yaxis_title="Cumulative Retention Uplift",
                )
                st.plotly_chart(fig_up, use_container_width=True)

        st.markdown("---")
        st.subheader("Optional NLP Extension — Dissatisfaction Theme Discovery")
        nlp = report.get("optional_nlp_insights", {})
        if nlp.get("available"):
            st.info(nlp["note"])
            n1, n2 = st.columns(2)
            with n1:
                st.markdown("**Discovered NMF Topics (TF-IDF)**")
                st.dataframe(pd.DataFrame(nlp["topics"]), use_container_width=True)
            with n2:
                reasons_df = pd.DataFrame(nlp["top_reasons"])
                fig_nlp = px.bar(
                    reasons_df[::-1],
                    x="count",
                    y="reason",
                    orientation="h",
                    title="Top Churn Reasons",
                )
                st.plotly_chart(fig_nlp, use_container_width=True)
        else:
            st.info(nlp.get("note", "No complaint text available for this dataset."))

        st.markdown("---")
        st.subheader("ML Engineering Metadata (MLflow & Database)")
        ml_eng = report.get("ml_engineering", {})
        if ml_eng:
            st.json(ml_eng)

    # ------------------------------------------------------------------
    # TAB 8: Customer Digital Twin, Live Prediction & CVM AI Agent
    # ------------------------------------------------------------------
    with tabs[7]:
        col_twin, col_agent = st.columns([1.15, 1.0])

        with col_twin:
            st.subheader("Customer Digital Twin & Live Prediction")
            st.caption(
                "Clone an existing subscriber (or score a brand-new profile) and adjust operational levers "
                "to simulate real-time changes in Churn Risk, CLV, Priority Score, and Next Best Offer."
            )
            id_col = report["dataset"]["id_col"]
            twin_options = ["<New Customer (Median Baseline)>"] + df[id_col].astype(str).head(100).tolist()
            twin_choice = st.selectbox("Select Twin Base", options=twin_options, key="twin_base")
            base_cid = None if twin_choice.startswith("<New") else twin_choice

            if base_cid:
                base_r = df[df[id_col].astype(str) == base_cid].iloc[0]
            else:
                base_r = df[report["dataset"]["feature_cols"]].median(numeric_only=True)

            overrides: Dict[str, Any] = {}
            if dataset_key == "iranian":
                c1, c2 = st.columns(2)
                overrides["complains"] = int(
                    c1.selectbox("Complaints (0=No, 1=Yes)", [0, 1], index=int(base_r.get("complains", 0)))
                )
                overrides["call_failure"] = int(
                    c2.slider("Call Failures", 0, 40, int(min(40, base_r.get("call_failure", 0))))
                )
                overrides["charge_amount"] = int(
                    c1.slider("Charge Level (0-10)", 0, 10, int(min(10, base_r.get("charge_amount", 1))))
                )
                overrides["subscription_length"] = int(
                    c2.slider("Subscription Length (months)", 1, 60, int(max(1, min(60, base_r.get("subscription_length", 12)))))
                )
                overrides["seconds_of_use"] = float(
                    st.slider("Seconds of Use", 0, 18000, int(min(18000, base_r.get("seconds_of_use", 3000))), step=100)
                )
            else:
                c1, c2 = st.columns(2)
                default_contract = (
                    "Two year"
                    if base_r.get("contract_two_year", 0) == 1
                    else ("One year" if base_r.get("contract_one_year", 0) == 1 else "Month-to-month")
                )
                overrides["contract_type"] = c1.selectbox(
                    "Contract Type",
                    ["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(default_contract),
                )
                overrides["tenure_months"] = int(
                    c2.slider("Tenure (Months)", 0, 72, int(min(72, base_r.get("tenure_months", 12))))
                )
                overrides["monthly_charges"] = float(
                    c1.slider("Monthly Charges ($)", 18.0, 120.0, float(min(120.0, max(18.0, base_r.get("monthly_charges", 65.0)))), step=1.0)
                )
                overrides["tech_support_yes"] = int(
                    c2.checkbox("Tech Support Add-on", value=bool(base_r.get("tech_support_yes", 0)))
                )
                overrides["online_security_yes"] = int(
                    c1.checkbox("Online Security Add-on", value=bool(base_r.get("online_security_yes", 0)))
                )

            sim_res = simulate_digital_twin(dataset_key, overrides=overrides, base_customer_id=base_cid)
            sim = sim_res["simulated"]
            deltas = sim_res["deltas"]

            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Twin Churn Risk",
                f"{sim['churn_prob']:.1%}",
                delta=f"{deltas['delta_churn_prob']:+.1%}" if deltas else None,
                delta_color="inverse",
            )
            k2.metric(
                "Twin Predicted Value",
                f"{sim['predicted_clv']:,.1f}",
                delta=f"{deltas['delta_clv']:+,.1f}" if deltas else None,
            )
            k3.metric(
                "Twin Priority Score",
                f"{sim['priority_score']:,.1f}",
                delta=f"{deltas['delta_priority']:+,.1f}" if deltas else None,
                delta_color="inverse",
            )
            st.success(
                f"**Simulated Next Best Offer:** {sim['recommended_offer']} "
                f"({sim['recommended_action']}) | Cost: `{sim['offer_cost']:.0f}` | "
                f"Sim. Net Benefit: `{sim['sim_expected_net_benefit']:,.1f}`"
            )

        with col_agent:
            st.subheader("CVM AI Agent (Retention Copilot)")
            st.caption(
                "Ask questions grounded in your live CVM models, SHAP drivers, segments, and Digital Twin."
            )
            sample_id = str(df.iloc[0][report["dataset"]["id_col"]])
            q_btn1, q_btn2 = st.columns(2)
            preset_q = None
            if q_btn1.button("Top 5 Priority Customers"):
                preset_q = "Show top 5 priority customers"
            if q_btn2.button(f"Explain {sample_id}"):
                preset_q = f"Explain customer {sample_id}"
            q_btn3, q_btn4 = st.columns(2)
            if q_btn3.button("Compare Segments"):
                preset_q = "Compare segments"
            if q_btn4.button(f"Digital Twin {sample_id}"):
                preset_q = f"Simulate digital twin for {sample_id}"

            user_q = st.text_input(
                "Ask the CVM Copilot",
                value=preset_q or "",
                placeholder=f"e.g., What if {sample_id} resolves complaints?",
            )
            if user_q:
                agent_res = ask_cvm_agent(dataset_key, user_q)
                st.markdown(agent_res["answer"])


if __name__ == "__main__":
    main()

