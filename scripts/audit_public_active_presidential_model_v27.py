"""Audit the public V27 pointer and all frozen rollback boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v27"
ROLLBACKS = {
    "v23": "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b",
    "v24": "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a",
    "v25": "218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b",
    "v26": "9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    for version, expected in ROLLBACKS.items():
        require(sha(ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv") == expected, f"{version} rollback drift")
    pointer = json.loads((ROOT / "data/config/current_presidential_model.json").read_text(encoding="utf-8"))
    require(pointer["active_version"] == "v27", "active pointer is not V27")
    require(pointer["runner"] == "scripts/run_active_presidential_model_v27.py", "runner drift")
    require(pointer["post_2022_outcomes_used"] is False, "outcome boundary drift")
    finalization = json.loads((ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8"))
    prediction = ACTIVE_DIR / "nested_predictions.csv"
    require(sha(prediction) == finalization["verification"]["v27_prediction_hash"], "V27 prediction drift")
    frame = pd.read_csv(prediction, low_memory=False)
    require(len(frame) == 232, "V27 row drift")
    require(not frame.election_id.astype(str).str.contains("2025").any(), "2025 leaked into history")
    require(np.allclose(frame.groupby(["election_id", "region_id"]).layer_pred.sum(), 1.0, atol=1e-10), "composition drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    require(summary["metrics"]["variant"] == "v27_core_weighted_party_regional_dispersion", "variant drift")
    require(np.isclose(summary["metrics"]["national_equal_election_macro_mae_pp"], 0.7209938807856883), "national level drift")
    require(summary["metrics"]["regional_equal_election_macro_mae_pp"] < 2.7122332621133673, "regional improvement lost")
    audit = pd.read_csv(ACTIVE_DIR / "party_regionalism_dispersion_audit.csv")
    require(set(audit.gain.round(8)) == {1.0}, "gain drift")
    intervals = json.loads((ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8"))
    require(intervals["model_version"] == "v27", "interval version drift")
    for record in finalization["artifacts"]:
        require(sha(ROOT / record["path"]) == record["sha256"], f"artifact drift: {record['path']}")
    print("[active V27 public audit: PASS]")
    print(f"v27_prediction_sha256={sha(prediction)}")
    print(f"regional_macro_pp={summary['metrics']['regional_equal_election_macro_mae_pp']}")
    print(f"national_macro_pp={summary['metrics']['national_equal_election_macro_mae_pp']}")


if __name__ == "__main__":
    main()
