"""Remove 2025 presidential rows from the reconstructed clean workspace.

This script is intentionally narrow: it rewrites only CSV files containing an
exact ``pres_2025`` field and records before/after hashes for every change.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "data",
    ROOT / "presidential_issue_engine" / "fixed_dataset",
)
AUDIT_PATH = ROOT / "docs" / "PRE2025_SANITIZATION_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_csv(path: Path) -> dict[str, object] | None:
    raw = path.read_bytes()
    if b"pres_2025" not in raw:
        return None

    before_hash = hashlib.sha256(raw).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    kept = [row for row in rows if "pres_2025" not in row]
    removed = len(rows) - len(kept)
    if removed == 0:
        return None

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(kept)

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
            change = sanitize_csv(path)
            if change is not None:
                changes.append(change)

    if not changes and AUDIT_PATH.exists():
        print("sanitized_files=0")
        print("removed_rows=0")
        print(f"audit_preserved={AUDIT_PATH}")
        return

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(
            {
                "policy": "remove every CSV row containing an exact pres_2025 field",
                "changed_files": changes,
                "changed_file_count": len(changes),
                "removed_row_count": sum(int(item["removed_rows"]) for item in changes),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"sanitized_files={len(changes)}")
    print(f"removed_rows={sum(int(item['removed_rows']) for item in changes)}")
    print(f"audit={AUDIT_PATH}")


if __name__ == "__main__":
    main()
