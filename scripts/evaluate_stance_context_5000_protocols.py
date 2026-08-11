"""Compare legacy and context labels on the same 1,000 rows per election."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import evaluate_neutral_context_protocols as protocol  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
PILOT_DIR = DATA_DIR / "protocol_pilots"
OUTPUT_DIR = DATA_DIR / "forecast_protocols"


def materialize_pilots() -> None:
    frame = pd.read_csv(DATA_DIR / "context_predictions_5000.csv", encoding="utf-8-sig")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    for election_id in protocol.ELECTIONS:
        election = frame.loc[frame["election_id"].eq(election_id)].copy()
        if len(election) != 1_000:
            raise RuntimeError(f"{election_id}: expected 1,000 rows, found {len(election)}")
        election.to_csv(PILOT_DIR / f"legacy_{election_id}.csv", index=False, encoding="utf-8-sig")
        context = election.copy()
        context["rule_stance_polarity"] = context["context_model_polarity"].astype(int)
        context["rule_stance_label"] = context["context_model_label"].map(
            {"negative": "attack", "neutral": "neutral", "positive": "endorse"}
        )
        context["rule_stance_confidence"] = context["context_model_probability"].astype(float)
        context.to_csv(PILOT_DIR / f"context_{election_id}.csv", index=False, encoding="utf-8-sig")


def signals_for(variant: str) -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == protocol.CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in protocol.ELECTIONS:
        features = build_features(config, pilot_input=PILOT_DIR / f"{variant}_{election_id}.csv")
        piece = features.loc[
            features["election_id"].eq(election_id),
            ["election_id", "slot", "stance_shadow_signal"],
        ].copy()
        piece.insert(0, "variant", variant)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def evaluate_variant(frame: pd.DataFrame, signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    signal_values = signals.loc[
        signals["variant"].eq(variant), ["election_id", "slot", "stance_shadow_signal"]
    ]
    rows = pd.concat(
        [
            protocol.full_fit_rows(frame, signal_values),
            protocol.loeo_rows(frame, signal_values),
            protocol.rolling_rows(frame, signal_values),
        ],
        ignore_index=True,
    )
    rows.insert(0, "variant", variant)
    return rows


def main() -> None:
    materialize_pilots()
    assembled = protocol.engine.assemble()
    frame = assembled.loc[assembled["election_id"].isin(protocol.ELECTIONS)].copy()
    signals = pd.concat([signals_for("legacy"), signals_for("context")], ignore_index=True)
    rows = pd.concat(
        [evaluate_variant(frame, signals, "legacy"), evaluate_variant(frame, signals, "context")],
        ignore_index=True,
    )
    row_summaries: list[pd.DataFrame] = []
    national_points: list[pd.DataFrame] = []
    for variant, group in rows.groupby("variant", sort=False):
        summary = protocol.row_summary(group.drop(columns="variant"))
        summary.insert(0, "variant", variant)
        row_summaries.append(summary)
        points = protocol.national_points(group.drop(columns="variant"))
        points.insert(0, "variant", variant)
        national_points.append(points)
    row_summary = pd.concat(row_summaries, ignore_index=True)
    points = pd.concat(national_points, ignore_index=True)
    national_summaries: list[pd.DataFrame] = []
    for variant, group in points.groupby("variant", sort=False):
        summary = protocol.national_summary(group.drop(columns="variant"))
        summary.insert(0, "variant", variant)
        national_summaries.append(summary)
    national_summary = pd.concat(national_summaries, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
