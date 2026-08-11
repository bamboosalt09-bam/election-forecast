"""Evaluate stance v3 primarily as an issue-weight overlay."""

from __future__ import annotations

import os
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
V3_DIR = DATA_DIR / "stance_v3"
OUTPUT_DIR = V3_DIR / "issue_overlay_protocols"
PILOT_DIR = OUTPUT_DIR / "pilot_inputs"
OVERLAY_PATH = V3_DIR / "stance_issue_overlay.csv"
DIRECT_SCALE = 0.10


def _assemble_with_overlay(overlay_path: Path | None) -> pd.DataFrame:
    previous = os.environ.get("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH")
    try:
        if overlay_path is not None:
            os.environ["POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"] = str(overlay_path)
        else:
            os.environ["POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"] = "off"
        frame = protocol.engine.assemble()
    finally:
        if previous is None:
            os.environ.pop("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", None)
        else:
            os.environ["POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"] = previous
    return frame.loc[frame["election_id"].isin(protocol.ELECTIONS)].copy()


def _overlay_ablation_paths() -> dict[str, Path]:
    source = pd.read_csv(OVERLAY_PATH, encoding="utf-8-sig")
    specifications = {
        "attention_only": (source["attention_multiplier"], 1.0),
        "character_only": (source["character_multiplier"], 1.0),
        "character_plus_link": (source["character_multiplier"], source["link_multiplier"]),
        "attention_character_no_link": (source["attention_character_multiplier"], 1.0),
        "attention_character_link": (
            source["attention_character_multiplier"],
            source["link_multiplier"],
        ),
    }
    paths: dict[str, Path] = {}
    for name, (salience, link) in specifications.items():
        ablation = source[["election_id", "issue_name", "slot", "available_date"]].copy()
        ablation["salience_multiplier"] = salience
        ablation["link_multiplier"] = link
        path = OUTPUT_DIR / f"overlay_{name}.csv"
        ablation.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    return paths


def _zero_signals(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = frame[["election_id", "slot"]].drop_duplicates().copy()
    out["stance_shadow_signal"] = 0.0
    out.insert(0, "variant", variant)
    return out


def _materialize_v3_inputs() -> None:
    frame = pd.read_csv(V3_DIR / "v3_predictions_5000.csv", encoding="utf-8-sig").fillna("")
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    for election_id in protocol.ELECTIONS:
        election = frame.loc[frame["election_id"].eq(election_id)].copy()
        election["rule_stance_label"] = election["v3_label"].map(
            {"negative": "attack", "neutral": "neutral", "positive": "endorse"}
        )
        election["rule_stance_polarity"] = election["v3_polarity"].astype(int)
        election["rule_stance_confidence"] = election["v3_max_probability"].astype(float)
        election.to_csv(
            PILOT_DIR / f"v3_{election_id}.csv", index=False, encoding="utf-8-sig"
        )


def _v3_signals() -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == protocol.CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in protocol.ELECTIONS:
        features = build_features(config, pilot_input=PILOT_DIR / f"v3_{election_id}.csv")
        piece = features.loc[
            features["election_id"].eq(election_id),
            ["election_id", "slot", "stance_shadow_signal"],
        ].copy()
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _evaluate(frame: pd.DataFrame, signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    selected = signals[["election_id", "slot", "stance_shadow_signal"]]
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
    _materialize_v3_inputs()
    original = _assemble_with_overlay(None)
    overlay_paths = _overlay_ablation_paths()
    overlay_frames = {
        name: _assemble_with_overlay(path) for name, path in overlay_paths.items()
    }
    overlay = overlay_frames["character_plus_link"]
    legacy = context_protocol.signals_for("legacy")
    v3 = _v3_signals()
    reduced_v3 = v3.copy()
    reduced_v3["stance_shadow_signal"] *= DIRECT_SCALE / protocol.SHADOW_SCALE

    variants = [
        ("baseline_no_stance", original, _zero_signals(original, "baseline_no_stance")),
        ("legacy_direct_060", original, legacy),
        ("v3_direct_010", original, reduced_v3),
        (
            "v3_issue_attention_only",
            overlay_frames["attention_only"],
            _zero_signals(overlay_frames["attention_only"], "v3_issue_attention_only"),
        ),
        (
            "v3_issue_character_only",
            overlay_frames["character_only"],
            _zero_signals(overlay_frames["character_only"], "v3_issue_character_only"),
        ),
        (
            "v3_issue_character_plus_link",
            overlay_frames["character_plus_link"],
            _zero_signals(
                overlay_frames["character_plus_link"], "v3_issue_character_plus_link"
            ),
        ),
        (
            "v3_issue_attention_character_no_link",
            overlay_frames["attention_character_no_link"],
            _zero_signals(
                overlay_frames["attention_character_no_link"],
                "v3_issue_attention_character_no_link",
            ),
        ),
        ("v3_issue_overlay", overlay, _zero_signals(overlay, "v3_issue_overlay")),
        ("v3_issue_overlay_legacy_direct060", overlay, legacy),
        ("v3_issue_overlay_direct010", overlay, reduced_v3),
    ]
    rows = pd.concat(
        [_evaluate(frame, signals, name) for name, frame, signals in variants],
        ignore_index=True,
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
        national = protocol.national_summary(points.drop(columns="variant"))
        national.insert(0, "variant", variant)
        national_summaries.append(national)
    row_summary = pd.concat(row_summaries, ignore_index=True)
    points = pd.concat(point_frames, ignore_index=True)
    national_summary = pd.concat(national_summaries, ignore_index=True)

    v3.to_csv(OUTPUT_DIR / "v3_candidate_signals_full_scale.csv", index=False, encoding="utf-8-sig")
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
            national_summary["protocol"].eq("rolling_origin")
            & ~national_summary["election_id"].eq("Overall"),
            ["variant", "election_id", "shadow_national_point_mae_pp"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
