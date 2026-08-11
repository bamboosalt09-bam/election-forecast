"""Build a deterministic hash manifest for the reconstructed clean baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "THROUGH2022_REDERIVED_MANIFEST.json"
ARCHIVE = Path(r"C:\english_folder\poll_project_post2025_outcome_aware_20260714")
ROOT_FILES = (".env.example", ".gitignore", "pyproject.toml", "README.md", "STRUCTURE.md")
SCAN_ROOTS = ("common", "data", "docs", "presidential_issue_engine", "scripts", "src", "tests")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if "report" in relative.parts and "through2022_rederived" not in relative.parts:
        return False
    return path.is_file()


def main() -> None:
    paths = [ROOT / name for name in ROOT_FILES if (ROOT / name).exists()]
    for name in SCAN_ROOTS:
        base = ROOT / name
        if base.exists():
            paths.extend(path for path in base.rglob("*") if included(path))
    unique = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in unique
    ]
    payload = {
        "baseline": "through2022_rederived",
        "information_cutoff": "2025-06-02",
        "archive_workspace": str(ARCHIVE),
        "archive_exists": ARCHIVE.exists(),
        "post2025_layers_default": "disabled",
        "file_count": len(files),
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest_files={len(files)}")
    print(f"manifest={OUTPUT}")


if __name__ == "__main__":
    main()
