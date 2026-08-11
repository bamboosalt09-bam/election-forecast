"""Evaluate a conservative target-aware stance resolver on the frozen 5,000 rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from election_forecast.stance_context_model import compose_context_input  # noqa: E402
from election_forecast.stance_target_policy import (  # noqa: E402
    generic_legacy_label,
    target_aware_decision,
)
from scripts import evaluate_neutral_context_protocols as protocol  # noqa: E402
from scripts import evaluate_stance_context_5000_protocols as context_protocol  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
OUTPUT_DIR = DATA_DIR / "target_aware_v2_protocols"
PILOT_DIR = OUTPUT_DIR / "pilot_inputs"
VARIANTS = {
    "target_v2_conservative": False,
    "target_v2_direct_override": True,
}


def _model_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    artifact = joblib.load(DATA_DIR / "stance_context_5000_v1.joblib")
    inputs = pd.Series([compose_context_input(row) for row in frame.to_dict(orient="records")])
    probabilities = artifact["model"].predict_proba(inputs.astype(str))
    classes = np.asarray(artifact["classes"])
    best_indices = np.argmax(probabilities, axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    return pd.DataFrame(
        {
            "model_label": classes[best_indices],
            "model_probability": probabilities[np.arange(len(frame)), best_indices],
            "model_margin": sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
        },
        index=frame.index,
    )


def apply_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    allow_override = VARIANTS[variant]
    model = _model_outputs(frame)
    output = frame.copy()
    output[["model_label", "model_probability", "model_margin"]] = model
    decisions = [
        target_aware_decision(
            row,
            model_label=str(row["model_label"]),
            model_probability=float(row["model_probability"]),
            model_margin=float(row["model_margin"]),
            allow_high_confidence_model_override=allow_override,
        )
        for row in output.to_dict(orient="records")
    ]
    output["target_v2_label"] = [decision.label for decision in decisions]
    output["target_v2_reason"] = [decision.reason for decision in decisions]
    output["target_v2_used_model_override"] = [
        int(decision.used_model_override) for decision in decisions
    ]
    legacy_generic = output["rule_stance_label"].map(generic_legacy_label)
    changed = ~output["target_v2_label"].eq(legacy_generic)
    output["target_v2_changed_from_legacy"] = changed.astype(int)

    output["aggregator_stance_label"] = output["rule_stance_label"].astype(str)
    output["aggregator_stance_polarity"] = pd.to_numeric(
        output["rule_stance_polarity"], errors="coerce"
    ).fillna(0).astype(int)
    output["aggregator_stance_confidence"] = pd.to_numeric(
        output["rule_stance_confidence"], errors="coerce"
    ).fillna(0.0)
    output.loc[changed, "aggregator_stance_label"] = output.loc[
        changed, "target_v2_label"
    ].map({"negative": "attack", "neutral": "neutral", "positive": "endorse"})
    output.loc[changed, "aggregator_stance_polarity"] = output.loc[
        changed, "target_v2_label"
    ].map({"negative": -1, "neutral": 0, "positive": 1})
    output.loc[changed, "aggregator_stance_confidence"] = output.loc[
        changed, "target_v2_label"
    ].map({"negative": 0.65, "neutral": 0.20, "positive": 0.65})
    output["rule_stance_label"] = output["aggregator_stance_label"]
    output["rule_stance_polarity"] = output["aggregator_stance_polarity"]
    output["rule_stance_confidence"] = output["aggregator_stance_confidence"]
    return output


def materialize_inputs() -> dict[str, pd.DataFrame]:
    source = pd.read_csv(DATA_DIR / "stance_context_5000.csv", encoding="utf-8-sig").fillna("")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        frame = apply_variant(source, variant)
        outputs[variant] = frame
        frame.to_csv(OUTPUT_DIR / f"{variant}_labels_5000.csv", index=False, encoding="utf-8-sig")
        for election_id in protocol.ELECTIONS:
            election = frame.loc[frame["election_id"].eq(election_id)].copy()
            if len(election) != 1_000:
                raise RuntimeError(f"{variant}/{election_id}: expected 1,000 rows")
            election.to_csv(
                PILOT_DIR / f"{variant}_{election_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )
    return outputs


def variant_signals(variant: str) -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == protocol.CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in protocol.ELECTIONS:
        features = build_features(
            config,
            pilot_input=PILOT_DIR / f"{variant}_{election_id}.csv",
        )
        piece = features.loc[
            features["election_id"].eq(election_id),
            ["election_id", "slot", "stance_shadow_signal"],
        ].copy()
        piece.insert(0, "variant", variant)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def evaluate_variant(frame: pd.DataFrame, signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    selected = signals.loc[
        signals["variant"].eq(variant), ["election_id", "slot", "stance_shadow_signal"]
    ]
    rows = pd.concat(
        [
            protocol.full_fit_rows(frame, selected),
            protocol.loeo_rows(frame, selected),
            protocol.rolling_rows(frame, selected),
        ],
        ignore_index=True,
    )
    rows.insert(0, "variant", variant)
    return rows


def classification_metrics() -> pd.DataFrame:
    gold = pd.read_csv(DATA_DIR / "gold_context_273.csv", encoding="utf-8-sig").fillna("")
    records: list[dict[str, object]] = []
    for variant in VARIANTS:
        resolved = apply_variant(gold.rename(columns={
            "stance_label": "rule_stance_label",
            "stance_polarity": "rule_stance_polarity",
            "stance_confidence": "rule_stance_confidence",
        }), variant)
        for scope, mask in (
            ("all_273", pd.Series(True, index=gold.index)),
            ("person_party_11", gold["target_type"].isin(["person", "party"])),
        ):
            truth = gold.loc[mask, "review_label"].astype(str)
            pred = resolved.loc[mask, "target_v2_label"].astype(str)
            records.append(
                {
                    "variant": variant,
                    "scope": scope,
                    "n": int(mask.sum()),
                    "accuracy": float(accuracy_score(truth, pred)),
                    "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                    "cohen_kappa": float(cohen_kappa_score(truth, pred)),
                    "warning": (
                        "development diagnostic; target rules were designed after inspecting "
                        "the 11 target gold rows"
                    ),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = materialize_inputs()
    classification = classification_metrics()
    legacy = context_protocol.signals_for("legacy").copy()
    legacy["variant"] = "legacy_rule"
    signals = pd.concat(
        [legacy, *[variant_signals(variant) for variant in VARIANTS]], ignore_index=True
    )

    assembled = protocol.engine.assemble()
    frame = assembled.loc[assembled["election_id"].isin(protocol.ELECTIONS)].copy()
    variants = signals["variant"].drop_duplicates().tolist()
    rows = pd.concat(
        [evaluate_variant(frame, signals, variant) for variant in variants], ignore_index=True
    )

    row_summaries: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    national_summaries: list[pd.DataFrame] = []
    for variant, group in rows.groupby("variant", sort=False):
        summary = protocol.row_summary(group.drop(columns="variant"))
        summary.insert(0, "variant", variant)
        row_summaries.append(summary)
        points = protocol.national_points(group.drop(columns="variant"))
        points.insert(0, "variant", variant)
        point_frames.append(points)
        national = protocol.national_summary(points.drop(columns="variant"))
        national.insert(0, "variant", variant)
        national_summaries.append(national)

    row_summary = pd.concat(row_summaries, ignore_index=True)
    points = pd.concat(point_frames, ignore_index=True)
    national_summary = pd.concat(national_summaries, ignore_index=True)
    reason_rows: list[dict[str, object]] = []
    for variant, labeled_frame in labeled.items():
        for reason, count in labeled_frame["target_v2_reason"].value_counts().items():
            reason_rows.append({"variant": variant, "reason": reason, "rows": int(count)})
    reasons = pd.DataFrame(reason_rows)

    classification.to_csv(OUTPUT_DIR / "classification_diagnostics.csv", index=False, encoding="utf-8-sig")
    reasons.to_csv(OUTPUT_DIR / "decision_reason_counts.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(OUTPUT_DIR / "candidate_signals.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(OUTPUT_DIR / "protocol_row_predictions.csv", index=False, encoding="utf-8-sig")
    row_summary.to_csv(OUTPUT_DIR / "protocol_row_summary.csv", index=False, encoding="utf-8-sig")
    points.to_csv(OUTPUT_DIR / "protocol_national_points.csv", index=False, encoding="utf-8-sig")
    national_summary.to_csv(
        OUTPUT_DIR / "protocol_national_summary.csv", index=False, encoding="utf-8-sig"
    )
    state = {
        "status": "complete",
        "variants": list(VARIANTS),
        "active_engine_changed": False,
        "classification_warning": "target-11 metrics are post-inspection development diagnostics",
    }
    (OUTPUT_DIR / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(classification.to_string(index=False))
    print()
    print(row_summary.loc[row_summary["election_id"].eq("Overall")].to_string(index=False))
    print()
    print(national_summary.loc[national_summary["election_id"].eq("Overall")].to_string(index=False))


if __name__ == "__main__":
    main()
