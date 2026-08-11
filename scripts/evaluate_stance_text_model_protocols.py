"""Evaluate quantitative-corrected stance pilots without changing active inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import evaluate_neutral_context_protocols as protocol  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


INPUT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_text_model_v1" / "corrected_pilots"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_text_model_v1" / "forecast_protocols"


def corrected_signals() -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == protocol.CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in protocol.ELECTIONS:
        features = build_features(config, pilot_input=INPUT_DIR / f"{election_id}.csv")
        pieces.append(
            features.loc[
                features["election_id"].eq(election_id),
                ["election_id", "slot", "stance_shadow_signal"],
            ].copy()
        )
    signals = pd.concat(pieces, ignore_index=True)
    if signals.duplicated(["election_id", "slot"]).any():
        raise RuntimeError("duplicate election-slot corrected signals")
    return signals


def main() -> None:
    assembled = protocol.engine.assemble()
    frame = assembled.loc[assembled["election_id"].isin(protocol.ELECTIONS)].copy()
    signals = corrected_signals()
    rows = pd.concat(
        [
            protocol.full_fit_rows(frame, signals),
            protocol.loeo_rows(frame, signals),
            protocol.rolling_rows(frame, signals),
        ],
        ignore_index=True,
    )
    row_metrics = protocol.row_summary(rows)
    points = protocol.national_points(rows)
    national_metrics = protocol.national_summary(points)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUTPUT_DIR / "candidate_signals.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(OUTPUT_DIR / "protocol_row_predictions.csv", index=False, encoding="utf-8-sig")
    row_metrics.to_csv(OUTPUT_DIR / "protocol_row_summary.csv", index=False, encoding="utf-8-sig")
    points.to_csv(OUTPUT_DIR / "protocol_national_points.csv", index=False, encoding="utf-8-sig")
    national_metrics.to_csv(OUTPUT_DIR / "protocol_national_summary.csv", index=False, encoding="utf-8-sig")
    print(row_metrics.to_string(index=False))
    print()
    print(national_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
