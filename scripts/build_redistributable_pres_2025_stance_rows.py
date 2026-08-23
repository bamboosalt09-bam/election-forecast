"""Derive a redistributable form of the 2025 Assembly stance rows.

The collected file
``data/raw/official_sources/assembly_pres_2025_minutes/assembly_stance_rows_2025_h1.csv``
carries ``text_excerpt`` - 48,588 verbatim excerpts from official proceedings,
each attributed to a named speaker. That is the sentence corpus this project
documents as excluded from redistribution, and it is why the 2025 demonstration
could not be rebuilt from the public tree.

Three places consume that file, and only one of them needs the words. Two take
the excerpt's length and nothing else. The third rematches issue keywords over
the text, which a length cannot stand in for - so this script publishes that
step's *output* rather than its input.

Two artifacts are emitted, both gzipped and both free of source text:

    assembly_stance_rows_2025_h1_public.csv.gz    the rows, text_length only
    assembly_issue_matches_2025_h1_public.csv.gz  the keyword rematch result

    35.0 MB with the corpus  ->  0.64 MB without it

The cost is stated rather than hidden: with these files a clean checkout
reproduces everything downstream of the keyword matching, and takes the matching
itself as given. docs/PRES_2025_INPUT_GUIDE.md describes how to obtain the
official proceedings and recompute the matching from scratch.

The ``stance_*`` and ``target_*`` columns are dropped as well. They are outputs
of the external stance models that V28 removed from the runtime, so the active
model does not read them and shipping them would reintroduce external-model
derivations the boundary exists to exclude.

Nothing here reconstructs the excerpts: only their character counts survive, and
``source_sha256`` stays so a holder of the official minutes can still verify
provenance row by row.

The output is deterministic - the gzip header carries no timestamp - so the
artifact hashes stably across rebuilds.
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

MINUTES_DIR = ROOT / "data/raw/official_sources/assembly_pres_2025_minutes"
SOURCE = MINUTES_DIR / "assembly_stance_rows_2025_h1.csv"
OUTPUT = MINUTES_DIR / "assembly_stance_rows_2025_h1_public.csv.gz"
#: Keyword matching over the excerpts cannot be reconstructed from their
#: lengths, so its *output* ships instead: one row per speech and issue. This is
#: the same class of artifact as the disclosed candidate-issue aggregate - a
#: derived table, no source sentence - and it is what lets a clean checkout
#: rebuild the 2025 forecast without the corpus.
MATCHES_OUTPUT = MINUTES_DIR / "assembly_issue_matches_2025_h1_public.csv.gz"

#: Exactly what the 2025 supplement path requires, plus the provenance hash.
REQUIRED_COLUMNS = (
    "election_id",
    "assembly_daesu",
    "source_file",
    "source_id",
    "source_row_id",
    "sentence_index",
    "meeting_date",
    "available_date",
    "period",
    "committee",
    "agenda",
    "speaker",
    "member_id",
    "issue_name",
    "issue_weight",
    "source_sha256",
)
DERIVED_COLUMN = "text_length"
WITHHELD_COLUMNS = ("text_excerpt", "stance_cues")


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"the collected stance rows are missing columns: {missing}")
    if "text_excerpt" not in frame.columns:
        raise ValueError("the collected stance rows carry no text_excerpt to measure")
    out = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    out[DERIVED_COLUMN] = frame["text_excerpt"].astype(str).str.len()
    for withheld in WITHHELD_COLUMNS:
        if withheld in out.columns:  # defensive: the point is that these never ship
            raise AssertionError(f"{withheld} must not reach the redistributable file")
    return out


def write_deterministic(frame: pd.DataFrame, destination: Path) -> str:
    payload = frame.to_csv(index=False, encoding="utf-8-sig", lineterminator="\n")
    raw = payload.encode("utf-8-sig") if isinstance(payload, str) else payload
    buffer = io.BytesIO()
    # mtime=0 and an empty filename keep the gzip header free of anything that
    # would change between rebuilds
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
        handle.write(raw)
    destination.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def derive_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the harness's own keyword rematch and keep only its result."""

    from scripts import run_prospective_forecast as harness

    matches, _diagnostics = harness._historical_compatible_target_matches(frame)
    if "text_excerpt" in matches.columns:
        raise AssertionError("the rematch output must not carry source text")
    return matches


def build(source: Path = SOURCE, destination: Path = OUTPUT) -> Path:
    if not source.is_file():
        raise SystemExit(
            f"the collected stance rows are not present: {source}\n"
            "This script derives the public file from the private collection; it "
            "cannot be run from a clean checkout."
        )
    frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    rows = derive(frame)
    digest = write_deterministic(rows, destination)
    print(destination.relative_to(ROOT).as_posix())
    print(f"  rows={len(rows)} bytes={destination.stat().st_size} sha256={digest}")

    matches = derive_matches(frame.fillna(""))
    matches_path = MATCHES_OUTPUT if destination == OUTPUT else destination.with_name(
        destination.name.replace("stance_rows", "issue_matches")
    )
    matches_digest = write_deterministic(matches, matches_path)
    print(matches_path.relative_to(ROOT).as_posix())
    print(
        f"  rows={len(matches)} bytes={matches_path.stat().st_size} "
        f"sha256={matches_digest}"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
