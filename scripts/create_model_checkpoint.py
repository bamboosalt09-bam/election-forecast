"""Create a restorable, hash-verified checkpoint before model experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "data/config/active_presidential_model.json",
    "data/raw/candidate_regional_base.csv",
    "data/raw/chungcheong_identity_alignment.csv",
    "data/raw/third_candidate_profile.csv",
    "data/raw/third_candidate_pressure.csv",
    "data/raw/mega_issue_intensity.csv",
    "data/raw/mega_issue_taxonomy.csv",
    "data/raw/election_generation_weights.csv",
    "data/raw/withdrawal_event_profiles.csv",
    "data/raw/withdrawn_candidate_transfers.csv",
    "data/raw/candidate_political_landscape.csv",
    "outputs/active_presidential_nested_v16",
    "outputs/automatic_contest_response_v10_ablation",
    "outputs/footprint_candidate_base_v9",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _copy_path(path: Path) -> str:
    """Use the Win32 long-path namespace for deeply nested checkpoints."""

    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--path", action="append", dest="paths")
    args = parser.parse_args()

    checkpoint_root = ROOT / "backups" / "model_checkpoints" / args.name
    temporary_root = checkpoint_root.with_name(checkpoint_root.name + ".partial")
    if checkpoint_root.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_root}")
    if temporary_root.exists():
        raise FileExistsError(f"partial checkpoint already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)

    requested = tuple(args.paths or DEFAULT_PATHS)
    records: list[dict[str, object]] = []
    for relative_text in requested:
        source = (ROOT / relative_text).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if ROOT not in source.parents and source != ROOT:
            raise ValueError(f"checkpoint source is outside workspace: {source}")
        for file_path in _files(source):
            relative = file_path.relative_to(ROOT)
            destination = temporary_root / "workspace" / relative
            os.makedirs(_copy_path(destination.parent), exist_ok=True)
            shutil.copy2(_copy_path(file_path), _copy_path(destination))
            records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": file_path.stat().st_size,
                    "sha256": _sha256(file_path),
                }
            )

    records.sort(key=lambda row: str(row["path"]))
    manifest = {
        "schema": "restorable_model_checkpoint_v1",
        "name": args.name,
        "created_at_local": datetime.now().astimezone().isoformat(),
        "workspace": str(ROOT),
        "post_2022_outcomes_used": False,
        "file_count": len(records),
        "files": records,
        "restore_policy": "copy workspace/ contents back only after explicit approval",
    }
    (temporary_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_root.replace(checkpoint_root)
    print(json.dumps({"checkpoint": str(checkpoint_root), "files": len(records)}))


if __name__ == "__main__":
    main()
