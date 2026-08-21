"""Behaviour guards for the raw-input inventory and readiness check."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import describe_inputs


def _write(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


@pytest.fixture()
def raw(tmp_path):
    """A miniature data/raw carrying one curated table and one source tree."""

    _write(
        tmp_path / "curated.csv",
        pd.DataFrame(
            {
                "election_id": ["pres_2017", "pres_2022"],
                "available_date": ["2017-04-01", "2022-02-01"],
                "value": [1.0, 2.0],
            }
        ),
    )
    _write(
        tmp_path / "shared_reference.csv",
        pd.DataFrame({"region_id": ["sido_11"], "weight": [1.0]}),
    )
    _write(
        tmp_path / "official_sources" / "cache" / "minutes.csv",
        pd.DataFrame(
            {
                "election_id": ["pres_2017"],
                # deliberately past the cutoff: downloaded material is filtered
                # by the loaders, so this must not be reported as a violation
                "available_date": ["2020-01-01"],
                "text": ["x"],
            }
        ),
    )
    return tmp_path


def test_scan_skips_downloaded_source_trees_unless_asked(raw) -> None:
    curated = {entry["path"] for entry in describe_inputs.scan(raw)}
    assert not any("official_sources" in path for path in curated)

    everything = {entry["path"] for entry in describe_inputs.scan(raw, include_sources=True)}
    assert any("official_sources" in path for path in everything)


def test_scan_separates_election_keyed_tables_from_shared_reference(raw) -> None:
    entries = {entry["path"].rsplit("/", 1)[-1]: entry for entry in describe_inputs.scan(raw)}
    assert entries["curated.csv"]["keyed_by_election"] is True
    assert entries["curated.csv"]["point_in_time"] is True
    assert entries["curated.csv"]["elections"] == ["pres_2017", "pres_2022"]
    assert entries["shared_reference.csv"]["keyed_by_election"] is False


def test_check_reports_a_covered_election_as_clean(raw) -> None:
    report = describe_inputs.check("pres_2017", cutoff="2017-05-08", raw=raw)
    assert [entry["path"].rsplit("/", 1)[-1] for entry in report["present"]] == [
        "curated.csv"
    ]
    assert report["missing"] == []
    assert report["point_in_time_violations"] == []


def test_check_names_the_gap_and_carries_its_schema(raw) -> None:
    report = describe_inputs.check("pres_2030", cutoff="2030-03-03", raw=raw)
    assert report["present"] == []
    assert len(report["missing"]) == 1
    gap = report["missing"][0]
    assert gap["path"].endswith("curated.csv")
    assert gap["columns"] == ["election_id", "available_date", "value"]


def test_check_blocks_on_a_curated_row_dated_after_the_cutoff(raw) -> None:
    """A curated row past D-1 would let the forecast read the future."""

    report = describe_inputs.check("pres_2017", cutoff="2017-01-01", raw=raw)
    violations = report["point_in_time_violations"]
    assert len(violations) == 1
    assert violations[0]["path"].endswith("curated.csv")
    assert violations[0]["rows_after_cutoff"] == 1


def test_downloaded_source_rows_past_the_cutoff_are_not_violations(raw) -> None:
    report = describe_inputs.check("pres_2017", cutoff="2017-01-01", raw=raw)
    assert not any(
        "official_sources" in entry["path"]
        for entry in report["point_in_time_violations"]
    )


def test_unregistered_election_requires_an_explicit_cutoff(raw) -> None:
    with pytest.raises(SystemExit):
        describe_inputs.check("pres_9999", raw=raw)


def test_registered_elections_resolve_to_their_d_minus_one_cutoff() -> None:
    cutoffs = describe_inputs.election_cutoffs()
    assert cutoffs["pres_2017"] == "2017-05-08"
    assert cutoffs["pres_2025"] == "2025-06-02"


def test_sources_counts_downloaded_rows_against_the_cutoff(raw) -> None:
    """The tree ``check`` ignores is exactly the one ``sources`` reports."""

    report = describe_inputs.sources("pres_2017", cutoff="2017-05-08", raw=raw)
    assert report["collected_rows"] == 1
    assert report["eligible_rows"] == 0  # the fixture row is dated 2020-01-01
    # the fixture is not under the repo, so paths report relative to it
    assert list(report["trees"]) == ["official_sources/cache"]

    later = describe_inputs.sources("pres_2017", cutoff="2021-01-01", raw=raw)
    assert later["eligible_rows"] == 1


def test_sources_agrees_with_the_recorded_2025_manifest() -> None:
    """The assembled 2025 minutes must reproduce the manifest's own counts."""

    report = describe_inputs.sources("pres_2025")
    minutes = report["trees"].get(
        "data/raw/official_sources/assembly_pres_2025_minutes"
    )
    if minutes is None:
        pytest.skip("assembled 2025 minutes are bulk source data, not tracked in git")
    assert minutes["collected"] == 48588
    assert minutes["eligible"] == 14985
