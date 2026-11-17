# Intelligent Customer Value Management (CVM) Platform

**Conception et développement d'une plateforme intelligente de gestion de la valeur client (CVM) basée sur l'apprentissage automatique, la prédiction du churn et les systèmes de recommandation personnalisés dans les télécommunications.**

An AI-driven predictive and personalized decision-support platform that unifies **Customer Segmentation & CLV Estimation**, **Churn Risk & SHAP Explainability**, **Content-Based & Learned Offer Acceptance Prediction**, **Utility-Based Next Best Action (NBA)**, **Customer Digital Twin Simulation**, and a **CVM AI Copilot**.

---

## 4-Module Connected Architecture

```text
DATA INTEGRATION LAYER (Source A Public + Source B Synthetic Campaigns + Source C Ooredoo-Ready)
        │
        ▼
FEATURE ENGINEERING & CUSTOMER 360 (RFM, Usage Intensity, Service Bundles, Leakage Removal)
        │
        ├──► Module 1: Customer Intelligence & Value Profiling (K-Means, GMM, Agglomerative, DBSCAN + CLV Regression)
        ├──► Module 2: Predictive Churn & Retention (LogReg, RF, XGBoost, LightGBM, CatBoost + 5-Fold CV + Calibration + SHAP)
        ├──► Module 3: Personalized Offer Recommendation (Content-Based Matching + Learned Offer Acceptance Model)
        │
        ▼
Module 4: Utility-Based Next Best Action (NBA) Engine
   Expected Utility(i, o) = P(Churn_i) × CLV_i × P(Accept_{i,o}) × Lift_o - Cost_o
        │
        ▼
FastAPI REST API + Streamlit CVM Intelligence Dashboard + Customer Digital Twin + AI Agent + MLflow + PostgreSQL
```

---

## Three-Source Data Strategy
1. **Source A — Real Public Telecom Datasets (Evaluated Separately)**:
   - `raw/iranian.csv` — UCI Iranian Telecom Churn (3,150 subscribers, 9-month behavioral window → month-12 churn + `Customer Value`).
   - `IBM_Telco_customer_churn_IBM_dataset.csv` — IBM Telco Customer Churn (7,043 subscribers, `CLTV`, `Churn Value`, `Churn Reason`; post-hoc leakage columns `Churn Score` and `Churn Reason` are excluded from modeling).
2. **Source B — Synthetic Telecom Business Data (`src/cvm/synthetic_campaign.py`)**:
   - Generates realistic campaign exposure logs, offer acceptance labels, and post-campaign retention/revenue outcomes (`artifacts/<dataset>/synthetic_campaigns.csv`) to train and evaluate Module 3 & Module 4. *Explicitly labeled as synthetic simulation data.*
3. **Source C — Ooredoo Internal Data (Schema-Ready)**:
   - Modular ingestion in `src/cvm/data_prep.py` + PostgreSQL persistence (`CVM_DATABASE_URL`) ready for anonymized/aggregated Ooredoo extracts.

---

## Quickstart

### Option 1: Docker Compose (Recommended for GitHub / Demo)
```bash
# 1. Run the ML pipeline once to populate artifacts/ (or mount precomputed artifacts)
docker compose up --build
```
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI Docs**: `http://localhost:8000/docs`
- **PostgreSQL**: `localhost:5432` (`cvm_db`)

### Option 2: Local Python + React Environment
```powershell
pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m cvm.pipeline --dataset all
uvicorn api.main:app --port 8000
```
In a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
- **React Decision-Support UI**: `http://localhost:5173`
- **FastAPI Swagger Docs**: `http://localhost:8000/docs`
