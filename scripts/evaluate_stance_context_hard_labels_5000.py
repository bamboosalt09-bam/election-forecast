"""Evaluate the 5,000 context predictions as hard three-value labels only.

The forecast input created here contains no classifier probabilities, margins,
or intensity columns. Positive and negative labels receive the same fixed
confidence solely because the existing aggregation API requires one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import evaluate_neutral_context_protocols as protocol  # noqa: E402
from scripts import evaluate_stance_context_5000_protocols as context_protocol  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
OUTPUT_DIR = DATA_DIR / "hard_label_forecast_protocols"
PILOT_DIR = OUTPUT_DIR / "pilot_inputs"
HARD_DIRECTION_CONFIDENCE = 0.65
HARD_NEUTRAL_CONFIDENCE = 0.20


def materialize_hard_inputs() -> None:
    source = pd.read_csv(DATA_DIR / "stance_context_5000.csv", encoding="utf-8-sig").fillna("")
    predictions = pd.read_csv(
        DATA_DIR / "context_predictions_5000.csv",
        encoding="utf-8-sig",
        usecols=["text_sha256", "context_model_label"],
    )
    if len(source) != 5_000 or source["text_sha256"].nunique() != 5_000:
        raise RuntimeError("source must contain 5,000 unique sentence hashes")
    if len(predictions) != 5_000 or predictions["text_sha256"].nunique() != 5_000:
        raise RuntimeError("predictions must contain 5,000 unique sentence hashes")
    frame = source.drop(
        columns=["rule_stance_label", "rule_stance_polarity", "rule_stance_confidence"],
        errors="ignore",
    ).merge(predictions, on="text_sha256", how="left", validate="one_to_one")
    if frame["context_model_label"].isna().any():
        raise RuntimeError("missing hard classifier labels after hash join")
    frame["hard_stance_label"] = frame["context_model_label"]
    frame["rule_stance_label"] = frame["hard_stance_label"].map(
        {"negative": "attack", "neutral": "neutral", "positive": "endorse"}
    )
    frame["rule_stance_polarity"] = frame["hard_stance_label"].map(
        {"negative": -1, "neutral": 0, "positive": 1}
    )
    frame["rule_stance_confidence"] = frame["hard_stance_label"].map(
        {
            "negative": HARD_DIRECTION_CONFIDENCE,
            "neutral": HARD_NEUTRAL_CONFIDENCE,
            "positive": HARD_DIRECTION_CONFIDENCE,
        }
    )
    frame = frame.drop(columns=["context_model_label"])
    forbidden = [
        column
        for column in frame.columns
        if "probability" in column or "strength" in column or "margin" in column
    ]
    if forbidden:
        raise RuntimeError(f"hard-label frame contains forbidden model scores: {forbidden}")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    for election_id in protocol.ELECTIONS:
        election = frame.loc[frame["election_id"].eq(election_id)].copy()
        if len(election) != 1_000:
            raise RuntimeError(f"{election_id}: expected 1,000 rows, found {len(election)}")
        election.to_csv(
            PILOT_DIR / f"hard_labels_{election_id}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    frame.to_csv(OUTPUT_DIR / "hard_labels_5000.csv", index=False, encoding="utf-8-sig")


def hard_signals() -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == protocol.CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in protocol.ELECTIONS:
        features = build_features(
            config,
            pilot_input=PILOT_DIR / f"hard_labels_{election_id}.csv",
        )
        piece = features.loc[
            features["election_id"].eq(election_id),
            ["election_id", "slot", "stance_shadow_signal"],
        ].copy()
        piece.insert(0, "variant", "context_hard_3class")
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    materialize_hard_inputs()
    legacy = context_protocol.signals_for("legacy").copy()
    legacy["variant"] = "legacy_rule"
    signals = pd.concat([legacy, hard_signals()], ignore_index=True)

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
        row_summary = protocol.row_summary(group.drop(columns="variant"))
        row_summary.insert(0, "variant", variant)
        row_summaries.append(row_summary)
        points = protocol.national_points(group.drop(columns="variant"))
        points.insert(0, "variant", variant)
        point_frames.append(points)
        national_summary = protocol.national_summary(points.drop(columns="variant"))
        national_summary.insert(0, "variant", variant)
        national_summaries.append(national_summary)

    row_summary = pd.concat(row_summaries, ignore_index=True)
    points = pd.concat(point_frames, ignore_index=True)
    national_summary = pd.concat(national_summaries, ignore_index=True)
    signals.to_csv(OUTPUT_DIR / "candidate_signals.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(OUTPUT_DIR / "protocol_row_predictions.csv", index=False, encoding="utf-8-sig")
    row_summary.to_csv(OUTPUT_DIR / "protocol_row_summary.csv", index=False, encoding="utf-8-sig")
    points.to_csv(OUTPUT_DIR / "protocol_national_points.csv", index=False, encoding="utf-8-sig")
    national_summary.to_csv(
        OUTPUT_DIR / "protocol_national_summary.csv", index=False, encoding="utf-8-sig"
    )

    print(row_summary.loc[row_summary["election_id"].eq("Overall")].to_string(index=False))
    print()
    print(national_summary.loc[national_summary["election_id"].eq("Overall")].to_string(index=False))
    print()
    print(
        national_summary.loc[
            national_summary["protocol"].isin(["loeo", "rolling_origin"])
            & ~national_summary["election_id"].eq("Overall"),
            ["variant", "protocol", "election_id", "shadow_national_point_mae_pp"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
