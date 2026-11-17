from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from api.main import app  # noqa: E402
from cvm.config import ARTIFACTS_DIR  # noqa: E402


def main() -> None:
    client = TestClient(app)

    res_health = client.get("/health")
    assert res_health.status_code == 200 and res_health.json()["status"] == "ok"

    res_ds = client.get("/datasets")
    assert res_ds.status_code == 200
    datasets = res_ds.json()
    assert all(d["ready"] for d in datasets), f"Not all datasets ready: {datasets}"

    for key in ["iranian", "ibm_telco"]:
        rep = client.get(f"/report/{key}")
        assert rep.status_code == 200
        data = rep.json()
        assert data["stage1_data_quality"]["clean_rows"] > 0

        custs = client.get(f"/customers/{key}?limit=5")
        assert custs.status_code == 200
        items = custs.json()["items"]
        assert len(items) == 5

        id_col = data["dataset"]["id_col"]
        cid = items[0][id_col]
        exp = client.get(f"/explain/{key}/{cid}")
        assert exp.status_code == 200
        exp_json = exp.json()
        assert len(exp_json["churn_shap_contributions"]) > 0
        assert len(exp_json["clv_shap_contributions"]) > 0

        up = client.get(f"/uplift/{key}")
        assert up.status_code == 200
        assert up.json().get("is_simulated") is True

        nlp = client.get(f"/nlp/{key}")
        assert nlp.status_code == 200

        twin = client.post(
            f"/predict/{key}",
            json={"base_customer_id": cid, "overrides": {}},
        )
        assert twin.status_code == 200
        assert "simulated" in twin.json()

        ag = client.post(
            f"/agent/{key}/ask",
            json={"query": f"Simulate digital twin for {cid}"},
        )
        assert ag.status_code == 200
        assert ag.json()["intent"] == "digital_twin_simulation"

        print(
            f"[OK] {key}: rows={data['stage1_data_quality']['clean_rows']}, "
            f"churn_best={data['stage4_churn_benchmark']['best_model']}, "
            f"clv_best={data['stage3_clv_benchmark']['best_model']}, "
            f"top_customer={cid} -> {exp_json['recommended_offer']}"
        )


if __name__ == "__main__":
    main()

