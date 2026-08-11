"""Run a deliberately weak, non-PIT raw-stance shadow experiment.

This script is not part of the active forecast engine. It uses the current
rule labels and meeting-date proxy only to test whether a future audited stance
layer could add signal. It never writes model inputs or headline metrics.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from presidential_issue_engine import issue_vote_engine as engine  # noqa: E402


STANCE_INPUT = ROOT / "outputs" / "assembly_stance" / "full_15_22" / "assembly_stance_rows_15_22.csv"
FEATURE_OUTPUT = ROOT / "outputs" / "assembly_stance" / "candidate_raw_stance_shadow.csv"
REPORT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "shadow_pres_2022" / "raw_stance_shadow_ab.md"
SCALE = 0.006  # Maximum intended direct pre-normalization movement is 0.6%p.
TARGET_WEIGHTS = {"person": 1.0, "party": 0.35}
STRICT_DIRECT_CONFIG = {
    "name": "direct_candidate_high_confidence",
    "target_weights": {"person": 1.0},
    "min_confidence": 0.65,
}


def candidate_reference() -> pd.DataFrame:
    results = pd.read_csv(ROOT / engine.RESULTS)
    active = results.loc[
        (results["slot"].astype(str) != "alpha")
        & results["is_active_slot"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    ][["election_id", "slot", "candidate_name", "party_name"]].drop_duplicates()
    return active.assign(
        candidate_name=lambda frame: frame["candidate_name"].fillna("").astype(str).str.strip(),
        party_name=lambda frame: frame["party_name"].fillna("").astype(str).str.strip(),
    )


def build_shadow_features(
    *,
    target_weights: dict[str, float] = TARGET_WEIGHTS,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    candidates = candidate_reference()
    person_map = {
        (str(row.election_id), str(row.candidate_name)): str(row.slot)
        for row in candidates.itertuples(index=False)
        if str(row.candidate_name).strip()
    }
    party_map = {
        (str(row.election_id), str(row.party_name)): str(row.slot)
        for row in candidates.itertuples(index=False)
        if str(row.party_name).strip()
    }
    totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, str, str]] = set()
    with STANCE_INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target_type = row.get("target_type", "")
            if target_type not in target_weights or row.get("stance_label") in {"", "neutral", "ambiguous"}:
                continue
            election_id = row.get("election_id", "")
            target = row.get("target_name", "")
            slot = (
                person_map.get((election_id, target))
                if target_type == "person"
                else party_map.get((election_id, target))
            )
            if not slot:
                continue
            key = (election_id, row.get("text_sha256", ""), target_type, target)
            if not key[1] or key in seen:
                continue
            seen.add(key)
            polarity = float(row.get("stance_polarity") or 0.0)
            confidence = float(row.get("stance_confidence") or 0.0)
            if confidence < min_confidence:
                continue
            issue_weight = float(row.get("issue_weight") or 0.0)
            weight = target_weights[target_type] * confidence * issue_weight
            cell = totals[(election_id, slot)]
            cell["signed"] += polarity * weight
            cell["absolute"] += abs(polarity) * weight
            cell["evidence_count"] += 1
            cell[f"{target_type}_evidence_count"] += 1

    out = candidates.copy()
    for column in ["signed", "absolute", "evidence_count", "person_evidence_count", "party_evidence_count"]:
        out[column] = [totals[(str(row.election_id), str(row.slot))][column] for row in out.itertuples(index=False)]
    # A four-unit pseudo-count and coverage cap keep this experimental signal weak.
    out["stance_net"] = out["signed"] / (out["absolute"] + 4.0)
    out["stance_coverage"] = np.minimum(np.sqrt(out["evidence_count"]) / 8.0, 1.0)
    out["stance_shadow_signal"] = out["stance_net"] * out["stance_coverage"]
    out["stance_shadow_signal"] -= out.groupby("election_id")["stance_shadow_signal"].transform("mean")
    out["availability_basis"] = "meeting_date_proxy_not_model_eligible"
    out["model_eligible"] = 0
    scope = ", ".join(sorted(target_weights))
    out["notes"] = (
        "Exploratory rule-label shadow only; "
        f"targets={scope}; min_confidence={min_confidence:.2f}; not PIT eligible"
    )
    return out[
        [
            "election_id", "slot", "candidate_name", "party_name", "signed", "absolute", "evidence_count",
            "person_evidence_count", "party_evidence_count", "stance_net", "stance_coverage",
            "stance_shadow_signal", "availability_basis", "model_eligible", "notes",
        ]
    ]


def _postprocess(frame: pd.DataFrame, train: pd.DataFrame, train_pred: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred = engine.apply_third_candidate_prediction_adjustment(frame, pred)
    pred = engine.apply_withdrawn_candidate_prediction_adjustment(frame, pred)
    pred = engine.apply_region_residual_calibration(train, frame, train_pred, pred)
    pred = engine.normalize_vote_share_predictions(frame, pred)
    for adjustment in (
        engine.apply_partisan_layer_prediction_moderation,
        engine.apply_party_context_prediction_adjustment,
        engine.apply_party_tone_gap_prediction_adjustment,
        engine.apply_same_orientation_external_adjustment,
        engine.apply_public_treatment_prediction_adjustment,
        engine.apply_generation_prediction_adjustment,
        engine.apply_candidate_conversion_context_adjustment,
        engine.apply_candidate_regionalism_adjustment,
    ):
        pred = adjustment(frame, pred)
    return pred


def _national_mae(frame: pd.DataFrame, pred: np.ndarray) -> float:
    out = frame[["slot", "votes"]].copy()
    out["actual"] = engine.normalized_vote_share_target(frame)
    out["pred"] = pred
    contest_votes = frame.groupby("region_id")["votes"].transform("sum").to_numpy(float)
    out["weight"] = contest_votes
    national = out.groupby("slot", as_index=False).apply(
        lambda group: pd.Series({
            "actual": np.average(group["actual"], weights=group["weight"]),
            "pred": np.average(group["pred"], weights=group["weight"]),
        }),
        include_groups=False,
    )
    return float((national["actual"] - national["pred"]).abs().mean() * 100)


def prepare_2022_ab() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = engine.assemble()
    order = {election_id: index for index, election_id in enumerate(engine.ORDER)}
    target = frame.loc[frame["election_id"].eq("pres_2022")].copy()
    train = frame.loc[frame["election_id"].map(order).lt(order["pres_2022"])].copy()
    X_train = train[engine.PREDICTORS].to_numpy(float)
    beta, _, _, _, means, scales = engine.ridge_fit(
        X_train, engine.normalized_vote_share_target(train), alpha=engine.RIDGE_ALPHA,
        sample_weight=engine.election_epoch_sample_weight(train),
    )
    train_raw = engine.ridge_predict(beta, X_train, means, scales)
    train_adjusted = engine.apply_withdrawn_candidate_prediction_adjustment(
        train, engine.apply_third_candidate_prediction_adjustment(train, train_raw)
    )
    baseline_raw = engine.ridge_predict(beta, target[engine.PREDICTORS].to_numpy(float), means, scales)
    baseline = _postprocess(target, train, train_adjusted, baseline_raw)
    actual = engine.normalized_vote_share_target(target)
    return target, baseline, actual


def run_2022_ab(
    features: pd.DataFrame,
    *,
    scale: float = SCALE,
    prepared: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    target, baseline, actual = prepared if prepared is not None else prepare_2022_ab()
    signals = target[["election_id", "slot"]].merge(
        features[["election_id", "slot", "stance_shadow_signal"]],
        on=["election_id", "slot"],
        how="left",
    )["stance_shadow_signal"]
    signals = pd.to_numeric(signals, errors="coerce").fillna(0.0).to_numpy(float)
    shadow = engine.normalize_vote_share_predictions(
        target, baseline + scale * signals
    )
    return {
        "row_mae_baseline_pp": float(np.abs(baseline - actual).mean() * 100),
        "row_mae_shadow_pp": float(np.abs(shadow - actual).mean() * 100),
        "national_mae_baseline_pp": _national_mae(target, baseline),
        "national_mae_shadow_pp": _national_mae(target, shadow),
        "mean_abs_shadow_shift_pp": float(np.abs(shadow - baseline).mean() * 100),
    }


def main() -> None:
    features = build_shadow_features()
    FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_OUTPUT, index=False, encoding="utf-8-sig")
    strict_features = build_shadow_features(
        target_weights=STRICT_DIRECT_CONFIG["target_weights"],
        min_confidence=STRICT_DIRECT_CONFIG["min_confidence"],
    )
    prepared = prepare_2022_ab()
    result = run_2022_ab(features, prepared=prepared)
    strict_result = run_2022_ab(strict_features, prepared=prepared)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = ["# 2022 Raw Stance Shadow A/B", "", "This is exploratory and non-PIT. It does not change the active engine.", ""]
    report.extend(["## Broad exact-target specification", ""])
    report.extend(f"- {key}: `{value:.4f}`" for key, value in result.items())
    report.extend(["", "## Strict direct-candidate specification", ""])
    report.extend(f"- {key}: `{value:.4f}`" for key, value in strict_result.items())
    report.extend(["", "## Feature Coverage: Broad", "", features.loc[features.election_id.eq("pres_2022")].to_csv(index=False)])
    report.extend(["", "## Feature Coverage: Strict", "", strict_features.loc[strict_features.election_id.eq("pres_2022")].to_csv(index=False)])
    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"broad": result, "strict": strict_result}, indent=2))
    print(f"features: {FEATURE_OUTPUT}")
    print(f"report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
