"""Answer two questions about the raw inputs without reading any model code.

    inventory              what tables exist and what each one is keyed on
    check <election_id>    what that election still needs before it can run

The engine reads fifty-odd CSV files under ``data/raw``. Which of them a new
target election actually needs is currently only discoverable by following the
loaders, so this module derives it from the files themselves: a table carrying
an ``election_id`` column is expected to carry rows for every election it is
used for, and a table that also carries ``available_date`` is subject to the
point-in-time cutoff at D-1.

Nothing here is election-specific, so the same command works for a future
target as for a scored one. ``check`` exits non-zero when it finds a row dated
after the cutoff, so it can gate a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ELECTION_COLUMN = "election_id"
DATE_COLUMN = "available_date"
# A table keyed by election that a scored election also leaves empty is shared
# reference data rather than a per-election requirement, so coverage is judged
# against elections that already run rather than against the full registry.
REFERENCE_ELECTIONS = ("pres_2017", "pres_2022")
# Downloaded source material is allowed to contain rows past the cutoff: the
# loaders filter it point-in-time on the way in, which is why the 2025 manifest
# records 14,985 eligible rows out of 48,588 collected. Only the curated tables
# a modeller maintains by hand are held to the cutoff directly, so the two are
# scanned as separate surfaces and only the curated one can block.
SOURCE_TREES = ("official_sources",)


def election_cutoffs() -> dict[str, str]:
    """Return the D-1 cutoff for every registered election."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from presidential_issue_engine.issue_vote_engine import ELECTION_DATES

    return {
        str(election): str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
        for election, date in ELECTION_DATES.items()
    }


def _read(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:  # noqa: BLE001 - an unparseable table is reported, not raised
        return None


def is_source_material(path: Path, raw: Path = RAW) -> bool:
    """True for downloaded source trees, which the loaders filter themselves."""

    return path.relative_to(raw).parts[0] in SOURCE_TREES


def _display(path: Path, raw: Path) -> str:
    """Repo-relative when the table lives in the repo, raw-relative otherwise.

    ``raw`` is overridable so the scan can be pointed at a fixture directory,
    which is not under ``ROOT`` and would otherwise fail to relativise.
    """

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.relative_to(raw).as_posix()


def scan(raw: Path = RAW, *, include_sources: bool = False) -> list[dict[str, object]]:
    """Describe every curated CSV under ``raw`` by what it is keyed on."""

    entries: list[dict[str, object]] = []
    for path in sorted(raw.rglob("*.csv")):
        source = is_source_material(path, raw)
        if source and not include_sources:
            continue
        frame = _read(path)
        relative = _display(path, raw)
        if frame is None:
            entries.append(
                {
                    "path": relative,
                    "absolute": str(path),
                    "unreadable": True,
                    "rows": None,
                    "source": source,
                }
            )
            continue
        keyed = ELECTION_COLUMN in frame.columns
        entries.append(
            {
                "path": relative,
                "absolute": str(path),
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "keyed_by_election": keyed,
                "point_in_time": DATE_COLUMN in frame.columns,
                "elections": (
                    sorted({str(value) for value in frame[ELECTION_COLUMN].dropna()})
                    if keyed
                    else []
                ),
                "source": source,
                "unreadable": False,
            }
        )
    return entries


def check(
    election_id: str,
    cutoff: str | None = None,
    raw: Path = RAW,
) -> dict[str, object]:
    """Report what ``election_id`` is missing, and where it reads the future.

    ``cutoff`` defaults to the day before the election. A table that the
    reference elections populate but this one does not is reported as a gap; a
    row dated after the cutoff is a point-in-time violation and always blocking.
    """

    if cutoff is None:
        cutoffs = election_cutoffs()
        if election_id not in cutoffs:
            raise SystemExit(
                f"{election_id} has no registered date; pass --cutoff YYYY-MM-DD"
            )
        cutoff = cutoffs[election_id]
    limit = pd.Timestamp(cutoff)

    present: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for entry in scan(raw):
        if entry.get("unreadable") or not entry.get("keyed_by_election"):
            continue
        elections = list(entry["elections"])
        covered = election_id in elections
        record = {"path": entry["path"], "point_in_time": bool(entry["point_in_time"])}
        if covered:
            present.append(record)
        elif any(reference in elections for reference in REFERENCE_ELECTIONS):
            # Carry the schema so the gap can be filled without opening the file.
            missing.append(dict(record, columns=list(entry["columns"])))
        if not (covered and entry["point_in_time"]):
            continue
        frame = _read(Path(str(entry["absolute"])))
        if frame is None:
            continue
        rows = frame.loc[frame[ELECTION_COLUMN].astype(str).eq(election_id)]
        dated = pd.to_datetime(rows[DATE_COLUMN], errors="coerce")
        late = int((dated > limit).sum())
        if late:
            violations.append(
                {
                    "path": entry["path"],
                    "rows_after_cutoff": late,
                    "latest": str(dated.max().date()),
                }
            )
    return {
        "election_id": election_id,
        "cutoff": str(limit.date()),
        "present": present,
        "missing": missing,
        "point_in_time_violations": violations,
    }


def sources(
    election_id: str,
    cutoff: str | None = None,
    raw: Path = RAW,
) -> dict[str, object]:
    """Summarise the downloaded material available for one election.

    ``check`` deliberately ignores these trees because the loaders filter them,
    but that leaves no way to see how much raw material an election actually
    has before a run consumes it. This reports collected rows against the rows
    that survive the cutoff, which is the number the run manifest records.
    """

    if cutoff is None:
        cutoffs = election_cutoffs()
        if election_id not in cutoffs:
            raise SystemExit(
                f"{election_id} has no registered date; pass --cutoff YYYY-MM-DD"
            )
        cutoff = cutoffs[election_id]
    limit = pd.Timestamp(cutoff)

    trees: dict[str, dict[str, int]] = {}
    collected = 0
    eligible = 0
    undated = 0
    for entry in scan(raw, include_sources=True):
        if not entry.get("source") or entry.get("unreadable"):
            continue
        if not entry.get("keyed_by_election") or election_id not in entry["elections"]:
            continue
        frame = _read(Path(str(entry["absolute"])))
        if frame is None:
            continue
        rows = frame.loc[frame[ELECTION_COLUMN].astype(str).eq(election_id)]
        total = int(len(rows))
        if entry["point_in_time"]:
            dated = pd.to_datetime(rows[DATE_COLUMN], errors="coerce")
            kept = int((dated <= limit).sum())
            blank = int(dated.isna().sum())
        else:
            # Without a date column the loaders cannot filter these rows here;
            # they are counted separately rather than assumed eligible.
            kept, blank = 0, total
        tree = Path(entry["path"]).parts
        key = "/".join(tree[: min(len(tree) - 1, 4)])
        bucket = trees.setdefault(key, {"files": 0, "collected": 0, "eligible": 0})
        bucket["files"] += 1
        bucket["collected"] += total
        bucket["eligible"] += kept
        collected += total
        eligible += kept
        undated += blank
    return {
        "election_id": election_id,
        "cutoff": str(limit.date()),
        "collected_rows": collected,
        "eligible_rows": eligible,
        "undated_rows": undated,
        "trees": trees,
    }


def _print_sources(report: dict[str, object]) -> None:
    print(f"{report['election_id']}   point-in-time cutoff {report['cutoff']}")
    print()
    trees = report["trees"]
    if not trees:
        print("No downloaded source rows are keyed to this election.")
        return
    for key in sorted(trees):
        bucket = trees[key]
        print(
            f"  {key:<52} {bucket['files']:>4} files"
            f"  {bucket['collected']:>7} rows  {bucket['eligible']:>7} eligible"
        )
    print()
    print(
        f"summed across trees: {report['collected_rows']} collected,"
        f" {report['eligible_rows']} within the cutoff,"
        f" {report['undated_rows']} carrying no date"
    )
    print(
        "Trees overlap - a cache holds the parts of the file assembled beside it -"
        " so read the per-tree rows, not the sum."
    )


def _print_inventory(entries: list[dict[str, object]]) -> None:
    keyed = [e for e in entries if e.get("keyed_by_election")]
    shared = [
        e for e in entries if not e.get("keyed_by_election") and not e.get("unreadable")
    ]
    broken = [e for e in entries if e.get("unreadable")]
    print(f"{len(entries)} tables under {RAW.relative_to(ROOT).as_posix()}")
    print()
    print(f"-- keyed by election_id ({len(keyed)}); PIT = carries available_date")
    for entry in keyed:
        flag = "PIT" if entry["point_in_time"] else "   "
        print(
            f"  {flag} {entry['path']:<56} {entry['rows']:>7} rows"
            f"  {len(entry['elections'])} elections"
        )
    print()
    print(f"-- shared reference, no election key ({len(shared)})")
    for entry in shared:
        print(f"      {entry['path']:<56} {entry['rows']:>7} rows")
    if broken:
        print()
        print(f"-- unreadable ({len(broken)})")
        for entry in broken:
            print(f"      {entry['path']}")


def _print_check(report: dict[str, object]) -> None:
    print(f"{report['election_id']}   point-in-time cutoff {report['cutoff']}")
    print()
    present = report["present"]
    print(f"-- already populated ({len(present)})")
    for entry in present:
        print(f"      {entry['path']}")
    missing = report["missing"]
    print()
    print(f"-- keyed tables with no rows for this election ({len(missing)})")
    for entry in missing:
        flag = "PIT" if entry["point_in_time"] else "   "
        print(f"  {flag} {entry['path']}")
        columns = [str(name) for name in entry.get("columns") or []]
        if columns:
            shown = ", ".join(columns[:8])
            extra = len(columns) - 8
            print(f"        columns: {shown}" + (f", (+{extra} more)" if extra > 0 else ""))
    if not report["present"] and missing:
        print()
        print(
            "None of these are populated. A forecast target is normally supplied\n"
            "through a generated context directory instead of data/raw - see\n"
            "scripts/run_prospective_forecast.py - so an empty list here means the\n"
            "election runs off that path, not that the run is misconfigured."
        )
    violations = report["point_in_time_violations"]
    print()
    print(f"-- rows dated after the cutoff ({len(violations)})")
    for entry in violations:
        print(
            f"      {entry['path']}: {entry['rows_after_cutoff']} rows,"
            f" latest {entry['latest']}"
        )
    if violations:
        print()
        print("A row dated after the cutoff lets the forecast read the future.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="describe every curated raw table")
    inventory.add_argument("--json", action="store_true")
    inventory.add_argument(
        "--include-sources",
        action="store_true",
        help="also list the downloaded source trees the loaders filter themselves",
    )
    checker = sub.add_parser("check", help="report what one election still needs")
    checker.add_argument("election_id")
    checker.add_argument("--cutoff", default=None, help="override the D-1 cutoff")
    checker.add_argument("--json", action="store_true")
    source = sub.add_parser(
        "sources", help="summarise downloaded material for one election"
    )
    source.add_argument("election_id")
    source.add_argument("--cutoff", default=None, help="override the D-1 cutoff")
    source.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.command == "inventory":
        entries = scan(include_sources=args.include_sources)
        if args.json:
            print(json.dumps(entries, ensure_ascii=False, indent=2))
        else:
            _print_inventory(entries)
        return

    if args.command == "sources":
        report = sources(args.election_id, args.cutoff)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_sources(report)
        return

    report = check(args.election_id, args.cutoff)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_check(report)
    if report["point_in_time_violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
