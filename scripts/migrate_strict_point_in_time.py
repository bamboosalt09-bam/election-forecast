"""Add auditable D-1 availability metadata to existing forecast aggregates.

This is a metadata migration over already extracted aggregate CSVs. It does not
read or reprocess raw Assembly transcripts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presidential_issue_engine.point_in_time import (  # noqa: E402
    cutoff_dates_as_strings,
    forecast_cutoff_map,
)


ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}
CUTOFFS = cutoff_dates_as_strings(ELECTION_DATES)
SALIENCE = ROOT / "data/issue_salience_assembly.csv"
LINK = ROOT / "data/candidate_issue_link.csv"
INTENSITY = ROOT / "data/raw/mega_issue_intensity.csv"
AXIS = ROOT / "data/raw/mega_issue_axis.csv"
GENERATION_WEIGHTS = ROOT / "data/raw/election_generation_weights.csv"
MANIFEST = ROOT / "data/raw/point_in_time_input_manifest.csv"


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def migrate_salience() -> None:
    frame = pd.read_csv(SALIENCE)
    period = pd.to_datetime(frame["period"], errors="coerce")
    cutoff = pd.to_datetime(frame["election_id"].map(CUTOFFS), errors="coerce")
    if period.isna().any() or cutoff.isna().any():
        raise ValueError("issue salience has invalid period or election metadata")
    frame["available_date"] = pd.concat(
        [period + pd.Timedelta(days=6), cutoff],
        axis=1,
    ).min(axis=1).dt.date.astype(str)
    _write(frame, SALIENCE)


def migrate_link() -> None:
    frame = pd.read_csv(LINK)
    frame["available_date"] = frame["election_id"].map(CUTOFFS)
    if frame["available_date"].isna().any():
        raise ValueError("candidate issue link has unknown elections")
    _write(frame, LINK)


def migrate_intensity() -> None:
    intensity = pd.read_csv(INTENSITY)
    axis = pd.read_csv(AXIS)
    axis["available_date"] = pd.to_datetime(axis["available_date"], errors="coerce")
    availability = axis.groupby("election_id")["available_date"].max()
    intensity["available_date"] = intensity["election_id"].map(availability)
    if intensity["available_date"].isna().any():
        raise ValueError("mega issue intensity lacks dated axis evidence")
    intensity["available_date"] = intensity["available_date"].dt.date.astype(str)
    _write(intensity, INTENSITY)


def migrate_generation_weights() -> None:
    frame = pd.read_csv(GENERATION_WEIGHTS)
    frame["available_date"] = frame["election_id"].map(CUTOFFS)
    if frame["available_date"].isna().any():
        raise ValueError("generation weights have unknown elections")
    _write(frame, GENERATION_WEIGHTS)


def write_manifest() -> None:
    paths = [
        SALIENCE,
        LINK,
        INTENSITY,
        ROOT / "data/raw/mega_issue_taxonomy.csv",
        GENERATION_WEIGHTS,
        AXIS,
        ROOT / "data/raw/issue_epoch_importance.csv",
        ROOT / "data/raw/issue_temporal_conversion.csv",
        ROOT / "data/raw/third_candidate_profile.csv",
        ROOT / "data/raw/third_candidate_pressure.csv",
        ROOT / "data/raw/candidate_regional_base.csv",
        ROOT / "data/raw/withdrawn_candidate_transfers.csv",
        ROOT / "data/raw/withdrawal_event_profiles.csv",
        ROOT / "data/raw/candidate_political_landscape.csv",
        ROOT / "data/raw/candidate_party_speech_context.csv",
        ROOT / "data/raw/candidate_party_tone_gap.csv",
        ROOT / "data/raw/candidate_public_treatment.csv",
        ROOT / "data/raw/candidate_vote_conversion_context.csv",
        ROOT / "data/raw/assembly_neutral_issue_context.csv",
        ROOT / "data/raw/candidate_generation_profile.csv",
        ROOT / "presidential_issue_engine/fixed_dataset/coalition_events.csv",
        ROOT / "presidential_issue_engine/fixed_dataset/scored_contest_scope.csv",
        ROOT / "presidential_issue_engine/fixed_dataset/economic_slot_alignment.csv",
        ROOT / "presidential_issue_engine/fixed_dataset/housing_slot_alignment.csv",
    ]
    cutoffs = forecast_cutoff_map(ELECTION_DATES)
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        available = pd.to_datetime(frame.get("available_date"), errors="coerce")
        election_cutoff = frame.get("election_id", pd.Series(index=frame.index, dtype=str)).astype(str).map(cutoffs)
        invalid = available.isna() | election_cutoff.isna() | available.gt(election_cutoff)
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "rows": len(frame),
                "min_available_date": available.min().date().isoformat() if available.notna().any() else "",
                "max_available_date": available.max().date().isoformat() if available.notna().any() else "",
                "invalid_point_in_time_rows": int(invalid.sum()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write(pd.DataFrame(rows), MANIFEST)


def main() -> None:
    migrate_salience()
    migrate_link()
    migrate_intensity()
    migrate_generation_weights()
    write_manifest()
    print(f"strict point-in-time metadata migrated; manifest={MANIFEST}")


if __name__ == "__main__":
    main()
