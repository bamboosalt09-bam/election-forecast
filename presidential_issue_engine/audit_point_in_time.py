"""Audit strict point-in-time controls for the presidential forecast engine.

The default audit validates dated input metadata and provenance hashes. Pass
``--deep`` to additionally mutate each target election outcome in a temporary
copy and prove that the target election predictor frame does not change.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import issue_vote_engine as engine
from presidential_issue_engine.build_assembly_speaker_influence import (
    clean_speaker_name,
    normalize_daesu,
)
from presidential_issue_engine.point_in_time import (
    filter_available_by_election,
    filter_observed_by_election,
)
from presidential_issue_engine.region_bloc_prior import election_date

MANIFEST = ROOT / "data/raw/point_in_time_input_manifest.csv"
ASSEMBLY_MATCHES = ROOT / "outputs/assembly_speaker_issue_matches_15_22.csv"
SPEAKER_PROFILE = ROOT / "data/raw/assembly_speaker_influence.csv"
ROSTER_15 = ROOT / "data/raw/assembly15_member_roster.csv"
ROSTER_ALL = ROOT / "data/assembly_roster.csv"
MEMBER_HISTORY = ROOT / "data/raw/assembly_member_history.csv"
ECONOMIC = ROOT / "presidential_issue_engine/fixed_dataset/economic_indicators.csv"
HOUSING = ROOT / "presidential_issue_engine/fixed_dataset/housing_price_index_sido.csv"
HOUSING_SGG = ROOT / "presidential_issue_engine/fixed_dataset/housing_price_index_sgg.csv"
KOSPI = ROOT / "presidential_issue_engine/fixed_dataset/kospi_daily.csv"
INTEREST_RATES = ROOT / "presidential_issue_engine/fixed_dataset/interest_rate_indicators.csv"
BLOC_HISTORY = ROOT / "presidential_issue_engine/fixed_dataset/bloc_history_results.csv"
RESULTS = ROOT / "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_manifest() -> dict[str, int]:
    _require(MANIFEST.exists(), f"Missing PIT manifest: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    required = {"path", "rows", "invalid_point_in_time_rows", "sha256"}
    _require(required.issubset(manifest.columns), "PIT manifest schema is incomplete")
    _require(
        pd.to_numeric(manifest["invalid_point_in_time_rows"], errors="coerce").fillna(1).eq(0).all(),
        "PIT manifest contains invalid rows",
    )
    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.path)
        _require(path.exists(), f"Manifest input is missing: {path}")
        _require(_sha256(path) == str(row.sha256), f"Manifest hash drift: {path}")
        frame = pd.read_csv(path)
        _require(len(frame) == int(row.rows), f"Manifest row-count drift: {path}")
        eligible = filter_available_by_election(
            frame,
            engine.ELECTION_DATES,
            source_name=str(row.path),
        )
        _require(len(eligible) == len(frame), f"Post-cutoff rows remain in active input: {path}")
    return {"manifest_files": len(manifest), "manifest_rows": int(manifest["rows"].sum())}


def audit_indicator_dates() -> dict[str, int]:
    total = 0
    for path in (ECONOMIC, HOUSING, HOUSING_SGG, KOSPI, INTEREST_RATES):
        frame = pd.read_csv(path)
        period_column = "period" if "period" in frame.columns else "date"
        _require({period_column, "available_date"}.issubset(frame.columns), f"Missing date columns: {path}")
        period = pd.to_datetime(frame[period_column], errors="coerce")
        available = pd.to_datetime(frame["available_date"], errors="coerce")
        _require(period.notna().all() and available.notna().all(), f"Invalid indicator dates: {path}")
        _require(available.ge(period).all(), f"Indicator published before observation period: {path}")
        total += len(frame)
    return {"indicator_rows": total}


def audit_bloc_history_dates() -> dict[str, int]:
    frame = pd.read_csv(BLOC_HISTORY)
    ids = frame["election_id"].dropna().astype(str).drop_duplicates()
    unknown = [value for value in ids if election_date(value) is None]
    _require(not unknown, f"Unknown election dates in bloc history: {unknown[:10]}")
    return {"bloc_history_rows": len(frame), "bloc_history_elections": len(ids)}


def audit_assembly_dates() -> dict[str, int]:
    if not ASSEMBLY_MATCHES.exists():
        return {"assembly_matches_present": 0, "assembly_match_rows": 0}
    frame = pd.read_csv(ASSEMBLY_MATCHES, usecols=["election_id", "meeting_date"])
    eligible = filter_observed_by_election(
        frame,
        engine.ELECTION_DATES,
        source_name="assembly_speaker_issue_matches",
        date_column="meeting_date",
    )
    _require(len(eligible) == len(frame), "Assembly matches contain post-cutoff rows")
    return {"assembly_matches_present": 1, "assembly_match_rows": len(frame)}


def audit_seniority_as_of() -> dict[str, int]:
    profile = pd.read_csv(
        SPEAKER_PROFILE,
        usecols=["assembly_daesu", "speaker_clean", "term_count"],
    )
    histories: list[pd.DataFrame] = []
    for path in (ROSTER_15, ROSTER_ALL, MEMBER_HISTORY):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"daesu", "name"}.issubset(frame.columns):
            histories.append(frame[["daesu", "name"]])
    _require(bool(histories), "No Assembly roster history available for seniority audit")
    history = pd.concat(histories, ignore_index=True).dropna()
    history["speaker_clean"] = history["name"].map(clean_speaker_name)
    history["term_daesu"] = pd.to_numeric(history["daesu"].map(normalize_daesu), errors="coerce")
    history = history.dropna(subset=["speaker_clean", "term_daesu"]).drop_duplicates(
        ["speaker_clean", "term_daesu"]
    )
    terms = history.groupby("speaker_clean")["term_daesu"].apply(tuple).to_dict()
    checked = 0
    inflated = 0
    for row in profile.itertuples(index=False):
        current = pd.to_numeric(row.assembly_daesu, errors="coerce")
        known_terms = terms.get(str(row.speaker_clean), ())
        if pd.isna(current) or not known_terms:
            continue
        expected = max(sum(term <= float(current) for term in known_terms), 1)
        checked += 1
        inflated += int(float(row.term_count) > expected)
    _require(inflated == 0, f"Future Assembly terms inflate {inflated} speaker rows")
    return {"seniority_rows_checked": checked}


def audit_target_outcome_invariance() -> dict[str, int]:
    original_results = engine.RESULTS
    original_order = list(engine.ORDER)
    original_regional_base_order = list(engine.REGIONAL_BASE_ORDER)
    checked = 0
    try:
        source = pd.read_csv(RESULTS)
        audit_order = list(original_order)
        engine.ORDER = audit_order
        engine.REGIONAL_BASE_ORDER = [*engine.WARMUP_ORDER, *audit_order]
        baseline = engine.assemble()
        for election_id in engine.ORDER:
            expected = baseline.loc[baseline["election_id"].eq(election_id)].sort_values(
                ["region_id", "slot"]
            )
            if expected.empty:
                continue
            mutated = source.copy()
            mask = mutated["election_id"].astype(str).eq(election_id)
            slots = mutated.loc[mask, "slot"].astype(str)
            slot_values = slots.map({"A": 0.05, "B": 0.15, "C": 0.80}).fillna(0.01)
            mutated.loc[mask, "vote_share"] = slot_values.to_numpy(float)
            mutated.loc[mask, "votes"] = (slot_values.to_numpy(float) * 10_000_000).astype(int)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
                temp_path = Path(handle.name)
            try:
                mutated.to_csv(temp_path, index=False, encoding="utf-8-sig")
                engine.RESULTS = str(temp_path)
                observed = engine.assemble()
                observed = observed.loc[observed["election_id"].eq(election_id)].sort_values(
                    ["region_id", "slot"]
                )
                _require(
                    expected[["region_id", "slot"]].reset_index(drop=True).equals(
                        observed[["region_id", "slot"]].reset_index(drop=True)
                    ),
                    f"Target row identity changed after outcome mutation: {election_id}",
                )
                audited_columns = [
                    *engine.PREDICTORS,
                    "assembly_neutral_issue_signal",
                    "assembly_neutral_issue_confidence",
                    "core_voting_mass",
                    "critical_voting_mass",
                    "swing_voting_mass",
                    *[
                        column
                        for column in expected.columns
                        if column.startswith("issue_pref_")
                        or column.startswith("issue_attention_")
                    ],
                ]
                left = expected[audited_columns].to_numpy(float)
                right = observed[audited_columns].to_numpy(float)
                _require(
                    np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True),
                    f"Target predictors or neutral issue signal depend on target outcome: {election_id}",
                )
                checked += len(expected)
            finally:
                temp_path.unlink(missing_ok=True)
    finally:
        engine.RESULTS = original_results
        engine.ORDER = original_order
        engine.REGIONAL_BASE_ORDER = original_regional_base_order
    return {"outcome_invariance_rows_checked": checked}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="also run target-outcome mutation checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, int] = {}
    for audit in (
        audit_manifest,
        audit_indicator_dates,
        audit_bloc_history_dates,
        audit_assembly_dates,
        audit_seniority_as_of,
    ):
        results.update(audit())
    if args.deep:
        results.update(audit_target_outcome_invariance())
    print("[strict PIT audit: PASS]")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
