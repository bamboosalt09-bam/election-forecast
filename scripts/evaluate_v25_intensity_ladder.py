"""Evaluate filling the mega-issue intensity ladder on the V25 stack.

The reachable intensities are the five values in ``SHOCK_CLASS_INTENSITY``, and
``intensity_activation`` is ``(intensity - 1).clip(0, 1)``, so a direct shock is
either inert or saturated: nothing between 1.00 and 2.00 can be reached. This
script measures a candidate that fills the gap from below, together with the
event-class alignment that the forecast path already applies and the
retrospective does not.

Design is a two-by-two so the two changes can be read apart:

    baseline            as promoted
    ladder              graded intensity only
    alignment           event-class alignment on the retrospective only
    ladder_alignment    both

Scope: this is a development comparison over the through-2022 sample. Every
prediction stays point-in-time safe, but the same five outcomes decide which
variant would be promoted, so the table is not an untouched holdout. No 2025
outcome is read anywhere.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from presidential_issue_engine.automatic_controls_v22 import (  # noqa: E402
    CRISIS_ACCOUNTABILITY,
    CRISIS_MIN_REGIME_EVIDENCE,
    SHOCK_CLASS_INTENSITY,
)

AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v23"
DIAGNOSTICS = (
    ROOT
    / "outputs"
    / "speech_derived_mega_intensity_v5"
    / "mega_issue_intensity_diagnostics.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "v25_intensity_ladder"
MIRROR_DIR = ROOT / "outputs" / "_v25_intensity_ladder_mirror"
CRISIS_INTENSITY = float(SHOCK_CLASS_INTENSITY["institutional_crisis"])
REGIME_COMPONENTS = ["salience_component", "severity_component", "breadth_component"]


@dataclass(frozen=True)
class Variant:
    name: str
    ladder: bool = False
    alignment: bool = False


VARIANTS = (
    Variant("baseline"),
    Variant("ladder", ladder=True),
    Variant("alignment", alignment=True),
    Variant("ladder_alignment", ladder=True, alignment=True),
)


def crisis_proximity(diagnostics: pd.DataFrame) -> pd.Series:
    """How close each election's evidence sits to the institutional-crisis gate.

    Exactly ``1.0`` for any election at or above both thresholds, so an election
    the classifier already calls a crisis is unaffected. Both thresholds are the
    classifier's own, so the measure introduces no new constant.
    """

    frame = diagnostics.copy()
    for column in REGIME_COMPONENTS + ["accountability_component"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    regime = frame[REGIME_COMPONENTS].min(axis=1) / CRISIS_MIN_REGIME_EVIDENCE
    accountability = frame["accountability_component"] / CRISIS_ACCOUNTABILITY
    return (regime.clip(0.0, 1.0) * accountability.clip(0.0, 1.0)).rename("proximity")


def ladder_intensity(intensity: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Raise each floor toward the crisis ceiling in proportion to proximity.

    The ceiling is the existing crisis level and each floor is that election's
    existing intensity, so an election already at the ceiling cannot move and an
    election with no crisis evidence keeps its current value.
    """

    proximity = dict(
        zip(
            diagnostics["election_id"].astype(str),
            crisis_proximity(diagnostics).astype(float),
        )
    )
    out = intensity.copy()
    floor = pd.to_numeric(out["mega_issue_intensity"], errors="coerce")
    weight = out["election_id"].astype(str).map(proximity).fillna(0.0)
    out["mega_issue_intensity"] = (floor + (CRISIS_INTENSITY - floor) * weight).round(6)
    return out


def _mirror(variant: Variant, diagnostics: pd.DataFrame) -> Path:
    if MIRROR_DIR.exists():
        shutil.rmtree(MIRROR_DIR)
    shutil.copytree(AUTOMATIC_DIR, MIRROR_DIR)
    if not variant.ladder:
        return MIRROR_DIR
    table = pd.read_csv(MIRROR_DIR / "mega_issue_intensity.csv", encoding="utf-8-sig")
    ladder_intensity(table, diagnostics).to_csv(
        MIRROR_DIR / "mega_issue_intensity.csv", index=False, encoding="utf-8-sig"
    )
    return MIRROR_DIR


def _metrics(predictions: pd.DataFrame, name: str) -> pd.DataFrame:
    column = (
        "candidate_name_x"
        if "candidate_name_x" in predictions.columns
        else "candidate_name"
    )
    rows: list[dict[str, object]] = []
    for election, group in predictions.groupby("election_id"):
        actual = group.groupby(column).apply(
            lambda frame: np.average(frame.actual, weights=frame.contest_votes)
        )
        predicted = group.groupby(column).apply(
            lambda frame: np.average(frame.layer_pred, weights=frame.contest_votes)
        )
        burdened = (
            group.groupby(column)["government_negative_share"].max().idxmax()
            if "government_negative_share" in group.columns
            else None
        )
        rows.append(
            {
                "variant": name,
                "election_id": str(election),
                "regional_mae_pp": float(
                    (group.layer_pred - group.actual).abs().mean() * 100
                ),
                "national_mae_pp": float((predicted - actual).abs().mean() * 100),
                "winner_correct": bool(predicted.idxmax() == actual.idxmax()),
                "burdened_candidate": str(burdened) if burdened else "",
                "burdened_error_pp": (
                    float((predicted[burdened] - actual[burdened]) * 100)
                    if burdened
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _evaluate(variant: Variant, diagnostics: pd.DataFrame) -> pd.DataFrame:
    from presidential_issue_engine import mega_issue_adjustment as mega
    from scripts import run_active_presidential_model_v25 as v25

    mirror = _mirror(variant, diagnostics)
    original_dir = v25.AUTOMATIC_DIR
    original_compile = mega.compile_direct_mega_scores
    taxonomy = pd.read_csv(mirror / "mega_issue_taxonomy.csv", encoding="utf-8-sig")

    def aligned(profile, intensity, election_dates, **kwargs):
        return original_compile(
            mega.align_profile_to_event_class(profile, taxonomy, election_dates),
            intensity,
            election_dates,
            **kwargs,
        )

    destination = OUTPUT_DIR / variant.name
    try:
        v25.AUTOMATIC_DIR = mirror
        if variant.alignment:
            mega.compile_direct_mega_scores = aligned
        v25.run(output_dir=destination)
    finally:
        v25.AUTOMATIC_DIR = original_dir
        mega.compile_direct_mega_scores = original_compile
        shutil.rmtree(MIRROR_DIR, ignore_errors=True)
    predictions = pd.read_csv(
        destination / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    return _metrics(predictions, variant.name)


def run(variants: tuple[Variant, ...] = VARIANTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(DIAGNOSTICS, encoding="utf-8-sig")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [_evaluate(variant, diagnostics) for variant in variants]
    by_election = pd.concat(frames, ignore_index=True)
    summary = (
        by_election.groupby("variant", sort=False)
        .agg(
            regional_macro_mae_pp=("regional_mae_pp", "mean"),
            national_macro_mae_pp=("national_mae_pp", "mean"),
            winners_correct=("winner_correct", "sum"),
            worst_burdened_error_pp=(
                "burdened_error_pp",
                lambda values: values.abs().max(),
            ),
        )
        .reset_index()
    )
    return summary, by_election


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.name for variant in VARIANTS],
        help="evaluate only the named variant; repeatable",
    )
    args = parser.parse_args()
    selected = (
        tuple(variant for variant in VARIANTS if variant.name in set(args.variant))
        if args.variant
        else VARIANTS
    )

    warnings.filterwarnings("ignore")
    diagnostics = pd.read_csv(DIAGNOSTICS, encoding="utf-8-sig")
    table = pd.read_csv(AUTOMATIC_DIR / "mega_issue_intensity.csv", encoding="utf-8-sig")
    ladder = ladder_intensity(table, diagnostics)
    print("intensity ladder")
    for election, before, after in zip(
        table["election_id"], table["mega_issue_intensity"], ladder["mega_issue_intensity"]
    ):
        activation = min(max(float(after) - 1.0, 0.0), 1.0)
        print(
            f"  {election:<10} {float(before):.2f} -> {float(after):.4f}"
            f"   activation {activation:.4f}"
        )
    print()

    summary, by_election = run(selected)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
