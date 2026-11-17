from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import ARTIFACTS_DIR

Base = declarative_base()


class ExperimentRun(Base):
    __tablename__ = "cvm_experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_key = Column(String(64), nullable=False, index=True)
    run_timestamp = Column(DateTime(timezone=True), nullable=False)
    row_count = Column(Integer, nullable=False)
    churn_rate = Column(Float, nullable=False)
    best_segmentation_model = Column(String(128), nullable=False)
    best_clv_model = Column(String(128), nullable=False)
    best_churn_model = Column(String(128), nullable=False)
    mlflow_run_id = Column(String(128), nullable=True)
    summary_json = Column(Text, nullable=False)


def get_database_url() -> str:
    """Return PostgreSQL URL from CVM_DATABASE_URL, or local SQLite fallback."""
    env_url = os.getenv("CVM_DATABASE_URL")
    if env_url:
        return env_url
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = (ARTIFACTS_DIR / "cvm_store.db").as_posix()
    return f"sqlite:///{db_path}"


def persist_pipeline_to_db(
    dataset_key: str,
    scored_df: pd.DataFrame,
    report: Dict[str, Any],
    mlflow_run_id: str | None = None,
) -> Dict[str, Any]:
    """Persist scored customer table and experiment run metadata to SQLAlchemy DB (PostgreSQL/SQLite)."""
    import json

    db_url = get_database_url()
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)

    # Save scored customers table per dataset
    table_name = f"scored_customers_{dataset_key}"
    scored_df.to_sql(table_name, engine, if_exists="replace", index=False)

    Session = sessionmaker(bind=engine, future=True)
    with Session() as session:
        run_row = ExperimentRun(
            dataset_key=dataset_key,
            run_timestamp=datetime.now(timezone.utc),
            row_count=int(report["stage1_data_quality"]["clean_rows"]),
            churn_rate=float(report["stage1_data_quality"]["churn_rate"]),
            best_segmentation_model=str(report["stage2_segmentation"]["best_model"]),
            best_clv_model=str(report["stage3_clv_benchmark"]["best_model"]),
            best_churn_model=str(report["stage4_churn_benchmark"]["best_model"]),
            mlflow_run_id=mlflow_run_id,
            summary_json=json.dumps(
                {
                    "clv_metrics": report["stage3_clv_benchmark"]["metrics_table"],
                    "churn_metrics": report["stage4_churn_benchmark"]["metrics_table"],
                }
            ),
        )
        session.add(run_row)
        session.commit()
        run_id = run_row.id

    backend = "postgresql" if db_url.startswith("postgresql") else "sqlite"
    return {
        "backend": backend,
        "database_url": db_url if backend == "sqlite" else "postgresql://***",
        "scored_table": table_name,
        "experiment_run_id": int(run_id),
    }

