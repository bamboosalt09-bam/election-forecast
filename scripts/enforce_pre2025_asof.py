"""Physically remove observations unavailable before the 2025 election.

The reconstructed baseline uses 2025-06-02 as its information cutoff. CSV
rows with a later ``available_date`` and dated lines in the raw KOSPI source
are removed. Every changed file is recorded with before/after hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = date(2025, 6, 2)
SCAN_ROOTS = (ROOT / "data", ROOT / "presidential_issue_engine" / "fixed_dataset")
KOSPI_SOURCE = ROOT / "data" / "raw" / "kospi_history_source.txt"
AUDIT_PATH = ROOT / "docs" / "PRE2025_ASOF_AUDIT.json"
KOREAN_DATE = re.compile(r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip()[:10])
    except (TypeError, ValueError):
        return None


def filter_csv(path: Path) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "available_date" not in reader.fieldnames:
            return None
        rows = list(reader)
        fieldnames = reader.fieldnames

    kept: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        available = parse_iso_date(row.get("available_date", ""))
        if available is not None and available > CUTOFF:
            removed += 1
        else:
            kept.append(row)
    if not removed:
        return None

    before_hash = sha256(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "removed_rows": removed,
        "sha256_before": before_hash,
        "sha256_after": sha256(path),
    }


def filter_kospi_source(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        match = KOREAN_DATE.match(line)
        observed = date(*map(int, match.groups())) if match else None
        if observed is not None and observed > CUTOFF:
            removed += 1
        else:
            kept.append(line)
    if not removed:
        return None

    before_hash = sha256(path)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "removed_rows": removed,
        "sha256_before": before_hash,
        "sha256_after": sha256(path),
    }


def main() -> None:
    changes: list[dict[str, object]] = []
    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.csv")):
            change = filter_csv(path)
            if change is not None:
                changes.append(change)
    source_change = filter_kospi_source(KOSPI_SOURCE)
    if source_change is not None:
        changes.append(source_change)

    if not changes and AUDIT_PATH.exists():
        print(f"cutoff={CUTOFF.isoformat()}")
        print("changed_files=0")
        print("removed_rows=0")
        print(f"audit_preserved={AUDIT_PATH}")
        return

    payload = {
        "policy": "remove observations with available_date after 2025-06-02",
        "cutoff_date": CUTOFF.isoformat(),
        "changed_files": changes,
        "changed_file_count": len(changes),
        "removed_row_count": sum(int(item["removed_rows"]) for item in changes),
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"cutoff={CUTOFF.isoformat()}")
    print(f"changed_files={len(changes)}")
    print(f"removed_rows={payload['removed_row_count']}")
    print(f"audit={AUDIT_PATH}")


if __name__ == "__main__":
    main()
