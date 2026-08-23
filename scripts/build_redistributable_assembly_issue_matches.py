"""Publish the historical Assembly issue matches the forecast path reads.

`scripts/build_speech_derived_issue_context.py` reads
``archives/experiments/manual_seed_lineage_v17_rejected_20260728/artifacts/
assembly_speaker_issue_matches_15_22.csv`` - 195,758 rows covering the five
scored elections. Two things are wrong with that as a shipped dependency:

* it sits under ``archives/``, which the repository boundary forbids tracking,
  and the directory name says it belongs to a *rejected* experiment even though
  the active forecast path depends on it;
* it is 118.9 MB, far past the tracked-file limit.

Neither is a rights problem. The file carries no verbatim text - it already
holds ``text_length`` rather than ``text_excerpt`` - so it is the same class of
derived table as the 2025 issue matches. The obstacle is purely size and
placement, and gzip settles the size: the same rows compress 39x, to 3.1 MB,
because ``committee``, ``agenda`` and ``source_file`` repeat heavily.

The output is deterministic - the gzip header carries no timestamp - so the
artifact hashes stably across rebuilds, and a differing hash means a differing
source rather than a flaky build.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = (
    ROOT
    / "archives/experiments/manual_seed_lineage_v17_rejected_20260728/artifacts"
    / "assembly_speaker_issue_matches_15_22.csv"
)
OUTPUT = (
    ROOT
    / "data/raw/official_sources"
    / "assembly_speaker_issue_matches_15_22.csv.gz"
)
SCORED_ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
WITHHELD_COLUMNS = ("text_excerpt", "stance_cues")


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    for withheld in WITHHELD_COLUMNS:
        if withheld in frame.columns:
            raise AssertionError(
                f"{withheld} must not reach the redistributable file; this source "
                "was expected to carry derived lengths only"
            )
    unexpected = sorted(set(frame["election_id"].dropna().unique()) - set(SCORED_ELECTIONS))
    if unexpected:
        raise ValueError(f"unexpected elections in the historical matches: {unexpected}")
    return frame


def write_deterministic(frame: pd.DataFrame, destination: Path) -> str:
    payload = frame.to_csv(index=False, encoding="utf-8-sig", lineterminator="\n")
    raw = payload.encode("utf-8-sig") if isinstance(payload, str) else payload
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9
    ) as handle:
        handle.write(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def build(source: Path = SOURCE, destination: Path = OUTPUT) -> Path:
    if not source.is_file():
        raise SystemExit(
            f"the archived issue matches are not present: {source}\n"
            "This script derives the published file from the archive; it cannot be "
            "run from a clean checkout."
        )
    frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    digest = write_deterministic(derive(frame), destination)
    print(destination.relative_to(ROOT).as_posix())
    print(f"  rows={len(frame)} bytes={destination.stat().st_size} sha256={digest}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
