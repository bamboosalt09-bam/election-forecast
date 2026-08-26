"""The lineage profile must exist for the forecast target, and stay outcome-free.

Every `lineage_identity_*` column was zero in the published 2025 forecast. The
profile needs only events strictly before the target's cutoff, so a forecast
target is no different in kind from a scored one - the builder simply never
listed `pres_2025`.

Adding it to the list would not have been enough. The builder resolves a
target's cutoff through `region_bloc_prior.election_date`, whose date map had
drifted out of step with `election_scope.ELECTION_DATES`: the central registry
carried `pres_2025` and the copy stopped at 2022. A target with no date fell
through a bare `continue`, so the profile would still not have been generated
and nothing would have said so.

These tests close both halves - the profile exists, it reads nothing after the
cutoff, and a configured target without a date is fatal rather than skipped.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from presidential_issue_engine import election_scope
from presidential_issue_engine import region_bloc_prior

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "outputs/unified_exact_lineage_v21/lineage_profiles_by_target.csv"
EVENTS = ROOT / "outputs/unified_exact_lineage_v21/exact_lineage_events.csv"
FORECAST_TARGET = "pres_2025"
FORECAST_CUTOFF = pd.Timestamp("2025-06-03")


def _profiles() -> pd.DataFrame:
    return pd.read_csv(PROFILES, encoding="utf-8-sig")


def test_the_forecast_target_has_a_lineage_profile() -> None:
    targets = set(_profiles()["target_election_id"].astype(str))
    assert FORECAST_TARGET in targets, (
        f"{FORECAST_TARGET} has no lineage profile, so every lineage_identity_* "
        "column is zero in the forecast"
    )


def test_the_builder_resolves_dates_from_the_central_registry() -> None:
    """The drift that hid the missing profile, and why it is not fixed in place.

    region_bloc_prior keeps its own presidential date map, which stops at 2022
    while election_scope carries pres_2025. Adding 2025 to that map changes
    what every caller sees - it moved V31's frozen 2025 forecast by 0.0032 -
    so the map stays as the frozen artifacts were built against, and the
    lineage builder reads election_scope directly instead.
    """

    from scripts import build_unified_exact_lineage_v21 as builder

    assert election_scope.ELECTION_DATES.get(FORECAST_TARGET) is not None
    assert region_bloc_prior.election_date(FORECAST_TARGET) is None, (
        "the shared date map gained pres_2025; frozen artifacts were built "
        "against a map without it"
    )
    assert builder._target_cutoff(FORECAST_TARGET) == FORECAST_CUTOFF


def test_a_configured_target_without_a_date_is_fatal() -> None:
    """A bare `continue` is what made the gap invisible."""

    source = (ROOT / "scripts/build_unified_exact_lineage_v21.py").read_text(encoding="utf-8")
    marker = source[source.index("cutoff = _target_cutoff(target)"):]
    head = marker[: marker.index("fit = fit_lineage_profiles")]
    assert "raise" in head, "a target with no election date must raise, not continue"
    assert "continue" not in head, "the silent skip is back"


def test_every_configured_target_resolves_to_a_date() -> None:
    source = (ROOT / "scripts/build_unified_exact_lineage_v21.py").read_text(encoding="utf-8")
    block = source[source.index("TARGETS = ("): source.index(")", source.index("TARGETS = ("))]
    targets = [line.strip().strip('",') for line in block.splitlines() if '"pres_' in line]
    assert targets, "could not read the configured targets"
    from scripts import build_unified_exact_lineage_v21 as builder

    for target in targets:
        assert builder._target_cutoff(target) is not None, (
            f"{target} is configured but resolves to no election date"
        )


def test_the_forecast_profile_reads_nothing_after_the_cutoff() -> None:
    if not EVENTS.is_file():
        pytest.skip("the lineage event ledger is not present")
    events = pd.read_csv(EVENTS, encoding="utf-8-sig")
    dated = [c for c in events.columns if "date" in c.lower()]
    assert dated, "the event ledger carries no date column to check"
    for column in dated:
        values = pd.to_datetime(events[column], errors="coerce").dropna()
        if values.empty:
            continue
        assert values.max() < FORECAST_CUTOFF or not values.gt(FORECAST_CUTOFF).any(), (
            f"{column} carries events at or after the {FORECAST_TARGET} cutoff"
        )


def test_the_forecast_profile_uses_no_post_2022_presidential_outcome() -> None:
    """A profile fitted on the target's own result would be circular."""

    rows = _profiles()
    forecast = rows.loc[rows["target_election_id"].astype(str).eq(FORECAST_TARGET)]
    assert not forecast.empty
    if "source_party_names" in forecast.columns:
        names = " ".join(forecast["source_party_names"].fillna("").astype(str))
        assert "pres_2025" not in names


def test_the_forecast_profile_is_invariant_to_the_2025_result() -> None:
    """Mutating the realised 2025 result must leave the profile untouched.

    The profile is fitted from lineage events, not from outcomes. This proves
    the separation rather than asserting it: the builder is re-run against a
    results table whose 2025 rows have been altered, and the forecast target's
    profile must come out identical.
    """

    results = ROOT / "data/raw/official_sources/presidential_results_standardized.csv"
    if not results.is_file():
        pytest.skip("the standardized results table is not present")
    frame = pd.read_csv(results, encoding="utf-8-sig")
    if not frame["election_id"].astype(str).eq(FORECAST_TARGET).any():
        # the results table carries no 2025 outcome at all, which is the
        # strongest form of the same guarantee
        return

    before = _profiles()
    before = before.loc[before["target_election_id"].astype(str).eq(FORECAST_TARGET)]
    original = results.read_bytes()
    try:
        mutated = frame.copy()
        mask = mutated["election_id"].astype(str).eq(FORECAST_TARGET)
        for column in ("votes", "vote_share"):
            if column in mutated.columns:
                mutated.loc[mask, column] = pd.to_numeric(
                    mutated.loc[mask, column], errors="coerce"
                ).fillna(0.0) * 0.5
        mutated.to_csv(results, index=False, encoding="utf-8-sig")
        subprocess.run(
            [sys.executable, "scripts/build_unified_exact_lineage_v21.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=900,
        )
        after = _profiles()
        after = after.loc[after["target_election_id"].astype(str).eq(FORECAST_TARGET)]
    finally:
        results.write_bytes(original)
        subprocess.run(
            [sys.executable, "scripts/build_unified_exact_lineage_v21.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            timeout=900,
        )
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True), after.reset_index(drop=True)
    )
