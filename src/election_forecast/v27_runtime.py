"""Verified access to the complete V27 runtime bundled in the wheel."""

from __future__ import annotations

import hashlib
from importlib.resources import as_file, files
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile


ARCHIVE_NAME = "_v27_runtime.zip"
MANIFEST_NAME = "_runtime_manifest.json"
EXPECTED_SCHEMA = "election_forecast_v27_packaged_runtime_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_parent() -> Path:
    override = os.environ.get("ELECTION_FORECAST_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "election-forecast"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "election-forecast"


def _read_manifest(archive: ZipFile) -> dict:
    manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    if manifest.get("schema") != EXPECTED_SCHEMA or manifest.get("active_version") != "v27":
        raise RuntimeError("packaged V27 runtime manifest is invalid")
    return manifest


def _verify_tree(root: Path, manifest: dict) -> None:
    for cache in root.rglob("__pycache__"):
        if cache.is_symlink():
            raise RuntimeError(f"symbolic link found in packaged V27 runtime: {cache}")
        if cache.is_dir():
            shutil.rmtree(cache)

    expected = {MANIFEST_NAME}
    for record in manifest["files"]:
        path = root / record["path"]
        expected.add(str(record["path"]).replace("\\", "/"))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"packaged V27 runtime file is missing: {record['path']}")
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"packaged V27 runtime file failed verification: {record['path']}")

    marker = root / MANIFEST_NAME
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError("packaged V27 runtime manifest marker is missing")
    if json.loads(marker.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("packaged V27 runtime manifest marker drifted")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symbolic link found in packaged V27 runtime: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise RuntimeError(
            f"packaged V27 runtime membership drift: extra={extra}, missing={missing}"
        )


def ensure_v27_runtime() -> Path:
    """Extract and verify the bundled runtime, returning its repository-like root."""

    resource = files("election_forecast").joinpath(ARCHIVE_NAME)
    if not resource.is_file():
        raise RuntimeError(
            "the V27 runtime archive is unavailable; reinstall from a built election-forecast wheel"
        )
    with as_file(resource) as archive_path:
        archive_digest = _sha256(archive_path)
        destination = _cache_parent() / "v27" / archive_digest[:16]
        marker = destination / MANIFEST_NAME
        with ZipFile(archive_path) as archive:
            manifest = _read_manifest(archive)
            if marker.is_file():
                _verify_tree(destination, manifest)
                return destination

            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f"{archive_digest[:16]}-", dir=destination.parent)
            )
            try:
                root = staging.resolve()
                for member in archive.infolist():
                    target = (staging / member.filename).resolve()
                    if root != target and root not in target.parents:
                        raise RuntimeError(f"unsafe path in V27 runtime archive: {member.filename}")
                archive.extractall(staging)
                _verify_tree(staging, manifest)
                if destination.exists():
                    _verify_tree(destination, manifest)
                else:
                    staging.replace(destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        return destination
