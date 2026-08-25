"""Audit active V30 and all frozen V23-V30 rollback boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v31"
ROLLBACKS = {
    "v23": "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b",
    "v24": "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a",
    "v25": "218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b",
    "v26": "9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3",
    "v27": "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b",
    "v28": "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba",
    "v29": "fed959cdba1e127f91c2ab640a378d1f44a4a3e79b4c4a76893cf8d7c6153904",
    "v30": "afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    for version, expected in ROLLBACKS.items():
        path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        require(sha(path) == expected, f"{version} rollback drift")
    pointer = json.loads(
        (ROOT / "data/config/current_presidential_model.json").read_text(encoding="utf-8")
    )
    require(pointer["active_version"] == "v31", "active pointer is not V31")
    require(pointer["post_2022_outcomes_used"] is False, "outcome boundary drift")

    frame = pd.read_csv(ACTIVE_DIR / "nested_predictions.csv", low_memory=False)
    require(len(frame) == 232, "V31 row drift")
    require(
        not frame.election_id.astype(str).str.contains("2025").any(),
        "2025 leaked into history",
    )
    require(
        np.allclose(
            frame.groupby(["election_id", "region_id"]).layer_pred.sum(), 1.0, atol=1e-10
        ),
        "composition drift",
    )
    require(bool((frame.layer_pred >= 0.0).all()), "the expansion produced a negative share")

    manifest = pd.read_csv(ACTIVE_DIR / "input_manifest.csv")
    normalized_paths = manifest.path.astype(str).str.replace("\\", "/", regex=False)
    require(
        not normalized_paths.str.contains(
            "assembly_issue_character_overlay", regex=False
        ).any(),
        "sentence-level external-model overlay remains active",
    )
    require(
        normalized_paths.str.endswith(
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ).any(),
        "disclosed frozen candidate-issue profile is missing",
    )
    require(
        not normalized_paths.str.endswith(
            ("mega_issue_axis.csv", "mega_issue_attribution.csv")
        ).any(),
        "unused external-model-derived seed remains active",
    )

    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    require(
        summary["metrics"]["variant"] == "v31_multiplicative_dispersion_expansion",
        "variant drift",
    )
    require(summary["external_neural_model_runtime"] is False, "neural runtime enabled")
    require(
        summary["external_model_derived_inputs"]
        == ["data/raw/auto_issue_seed/candidate_issue_profile.csv"],
        "retained derived-input disclosure drift",
    )

    # No comparison against the predecessor's score belongs in an audit. V30
    # carried `national <= V29` and `regional <= V29`, which would have rejected
    # its own subject had the recorded +0.0119pp cost materialised - and would
    # reject V31, whose national figure is 0.0039pp worse on purpose. A version
    # is right or wrong about what it reads and what it can produce, not about
    # whether the panel liked it.
    #
    # What is checked instead: the published figures are the ones the artifact
    # produces, and the transform cannot emit the value that motivated V31.
    require(
        abs(summary["metrics"]["regional_equal_election_macro_mae_pp"]
            - float(pointer["regional_equal_election_macro_mae_pp"])) < 1e-12,
        "the pointer's regional macro disagrees with the artifact",
    )
    require(
        abs(summary["metrics"]["national_equal_election_macro_mae_pp"]
            - float(pointer["national_equal_election_macro_mae_pp"])) < 1e-12,
        "the pointer's national macro disagrees with the artifact",
    )
    require(
        float(frame["layer_pred"].min()) > 0.0,
        "a share reached zero; the multiplicative form cannot do that from a "
        "positive input, so the transform did not run",
    )
    expansion_audit = pd.read_csv(
        ACTIVE_DIR / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    require(
        "feasibility_capped" not in expansion_audit.columns,
        "a feasibility cap reappeared; V31 removed the condition that needed one",
    )
    require(
        bool((expansion_audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()),
        "the level reconciliation did not hold the candidate levels",
    )
    require(
        bool((expansion_audit["reconciliation_rounds"] >= 1).all()),
        "the reconciliation did not run",
    )
    # the target election's own turnout must not reach either transform
    require(
        "forecast_time_region_weight" in frame.columns,
        "the forecast-time weight column is absent from the artifact",
    )
    require(
        bool(
            (
                (frame["layer_pred"] - frame["actual"]) * 100 - frame["err_pp"]
            ).abs().max()
            < 1e-9
        ),
        "err_pp does not describe the shipped prediction",
    )

    expansion = pd.read_csv(
        ACTIVE_DIR / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    require(
        bool((expansion["max_candidate_level_shift_pp"].abs() < 1e-9).all()),
        "a candidate national level moved by more than machine noise",
    )
    untouched = expansion.loc[expansion.election_id.eq("pres_2012")]
    require(
        float(untouched.iloc[0]["expansion_factor"]) == 1.0,
        "the two-candidate election was not left alone",
    )

    finalization = json.loads(
        (ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8")
    )
    require(
        sha(ACTIVE_DIR / "nested_predictions.csv")
        == finalization["verification"]["v31_prediction_hash"],
        "V31 prediction drift",
    )
    for record in finalization["artifacts"]:
        require(sha(ROOT / record["path"]) == record["sha256"], f"artifact drift: {record['path']}")
    print("[active V31 public audit: PASS]")
    print(f"v31_prediction_sha256={sha(ACTIVE_DIR / 'nested_predictions.csv')}")


if __name__ == "__main__":
    main()
