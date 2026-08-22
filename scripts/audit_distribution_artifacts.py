"""Audit source and wheel distributions before publication.

The check is intentionally independent of setuptools.  It opens the produced
archives directly, rejects unsafe or non-public members, and verifies the
hash-indexed V28 runtime embedded in the wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from zipfile import ZipFile


FROZEN_V27_SHA256 = (
    "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b"
)
FORBIDDEN_FILES = {
    "presidential_issue_engine/fixed_dataset/kospi_daily.csv",
    "data/raw/kospi_history_source.txt",
    "data/raw/official_sources/assembly_pres_2025_minutes/assembly_stance_rows_2025_h1.csv",
    "data/raw/assembly_issue_character_overlay.csv",
    "data/raw/auto_issue_seed/candidate_issue_profile.csv",
    "data/raw/auto_issue_seed/mega_issue_axis.csv",
    "data/raw/auto_issue_seed/mega_issue_attribution.csv",
}
FORBIDDEN_PREFIXES = (
    "data/cache/",
    "data/generated/",
    "data/imports/",
    "data/logs/",
    "data/news_analyzed/",
    "data/news_cleaned/",
    "data/news_raw/",
    "data/raw_lake/",
    "data/shadow/",
    "data/raw/official_sources/cache/",
    "data/raw/official_sources/checkpoints/",
    "presidential_issue_engine/report/through2022_rederived/overlay_variants/",
)
FORBIDDEN_WHEEL_MODULE_PREFIX = "election_forecast/stance_"
RESEARCH_DISTRIBUTION_PREFIXES = (
    "research/",
    "outputs/automatic_controls_v23_ablation_v3/",
    "outputs/prospective_pres_2025_v23/",
    "outputs/prospective_pres_2025_v24/",
    "outputs/prospective_pres_2025_v25/",
    "outputs/v24_defect_ablation/",
    "outputs/v24_floor_recalibration_hypotheses/",
    "outputs/v24_structural_residual_hypotheses/",
)
REQUIRED_RUNTIME_FILES = {
    "LICENSE",
    "NOTICE",
    "scripts/run_current_presidential_model.py",
    "scripts/run_active_presidential_model_v28.py",
    "scripts/run_prospective_forecast_v28.py",
    "scripts/audit_public_active_presidential_model_v28.py",
    "scripts/verify_v28_clean_reproduction.py",
    "presidential_issue_engine/issue_vote_engine.py",
    "outputs/active_presidential_nested_v28/nested_predictions.csv",
}


def _safe_member(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise RuntimeError(f"unsafe distribution member: {name}")
    return path.as_posix()


def _without_distribution_root(name: str) -> str:
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else name


def _assert_public(name: str, *, strip_distribution_root: bool = False) -> None:
    relative = _safe_member(name)
    if strip_distribution_root:
        relative = _without_distribution_root(relative)
    if relative in FORBIDDEN_FILES or relative.startswith(FORBIDDEN_PREFIXES):
        raise RuntimeError(f"non-public file included in distribution: {name}")
    if relative.startswith(RESEARCH_DISTRIBUTION_PREFIXES):
        raise RuntimeError(f"research-only artifact included in distribution: {name}")
    if "/__pycache__/" in f"/{relative}" or relative.endswith((".pyc", ".pyo")):
        raise RuntimeError(f"generated Python cache included in distribution: {name}")


def _audit_runtime(payload: bytes) -> int:
    with ZipFile(io.BytesIO(payload)) as runtime:
        names = [_safe_member(name) for name in runtime.namelist()]
        for name in names:
            _assert_public(name)
            if name.startswith(FORBIDDEN_WHEEL_MODULE_PREFIX) and name.endswith(".py"):
                raise RuntimeError(f"inactive external-model module included in wheel: {name}")
        if "_runtime_manifest.json" not in names:
            raise RuntimeError("wheel runtime manifest is missing")
        manifest = json.loads(runtime.read("_runtime_manifest.json"))
        if manifest.get("active_version") != "v28":
            raise RuntimeError("wheel runtime does not declare active V28")
        if manifest.get("source_boundary") != "git-tracked-public-files-only":
            raise RuntimeError("wheel runtime has an unexpected source boundary")
        if manifest.get("post_2022_outcomes_used") is not False:
            raise RuntimeError("wheel runtime does not preserve the outcome-free boundary")
        if manifest.get("frozen_prediction_sha256") != FROZEN_V27_SHA256:
            raise RuntimeError("wheel runtime frozen V27 hash declaration drifted")

        records = manifest.get("files")
        if not isinstance(records, list):
            raise RuntimeError("wheel runtime file manifest is invalid")
        indexed: dict[str, dict[str, object]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise RuntimeError("wheel runtime contains an invalid file record")
            relative = _safe_member(str(record["path"]))
            _assert_public(relative)
            if relative in indexed:
                raise RuntimeError(f"duplicate wheel runtime file record: {relative}")
            indexed[relative] = record

        missing = sorted(REQUIRED_RUNTIME_FILES - indexed.keys())
        if missing:
            raise RuntimeError(f"wheel runtime is incomplete: {missing}")
        archive_payloads = set(names) - {"_runtime_manifest.json"}
        if archive_payloads != set(indexed):
            raise RuntimeError("wheel runtime payload and manifest membership differ")
        for relative, record in indexed.items():
            content = runtime.read(relative)
            if int(record.get("bytes", -1)) != len(content):
                raise RuntimeError(f"wheel runtime size mismatch: {relative}")
            if record.get("sha256") != hashlib.sha256(content).hexdigest():
                raise RuntimeError(f"wheel runtime hash mismatch: {relative}")
        frozen = runtime.read(
            "outputs/active_presidential_nested_v28/nested_predictions.csv"
        )
        if hashlib.sha256(frozen).hexdigest() != FROZEN_V27_SHA256:
            raise RuntimeError("embedded frozen V27 prediction hash drifted")
        return len(indexed)


def audit_wheel(path: Path) -> tuple[int, int]:
    with ZipFile(path) as wheel:
        names = [_safe_member(name) for name in wheel.namelist()]
        for name in names:
            _assert_public(name)
        runtime_names = [
            name for name in names if name.endswith("election_forecast/_v28_runtime.zip")
        ]
        if len(runtime_names) != 1:
            raise RuntimeError(
                f"wheel must contain exactly one V28 runtime archive: {path.name}"
            )
        runtime_files = _audit_runtime(wheel.read(runtime_names[0]))
        return len(names), runtime_files


def audit_sdist(path: Path) -> int:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError(f"empty source distribution: {path.name}")
        roots: set[str] = set()
        for member in members:
            name = _safe_member(member.name)
            roots.add(PurePosixPath(name).parts[0])
            _assert_public(name, strip_distribution_root=True)
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"unsafe linked/device member in source distribution: {name}")
        if len(roots) != 1:
            raise RuntimeError(f"source distribution has multiple top-level roots: {roots}")
        relative_names = {_without_distribution_root(_safe_member(item.name)) for item in members}
        required = {
            "LICENSE",
            "NOTICE",
            "SECURITY.md",
            "requirements-v27.lock",
            "setup.py",
            "scripts/audit_public_data_rights.py",
            "scripts/audit_publication_security.py",
        }
        missing = sorted(required - relative_names)
        if missing:
            raise RuntimeError(f"source distribution is incomplete: {missing}")
        return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    wheels = sorted(args.dist_dir.glob("*.whl"))
    if not sdists or not wheels:
        raise RuntimeError("distribution audit requires at least one sdist and one wheel")

    sdist_members = sum(audit_sdist(path) for path in sdists)
    wheel_members = 0
    runtime_files = 0
    for path in wheels:
        outer, inner = audit_wheel(path)
        wheel_members += outer
        runtime_files += inner

    print("[distribution artifact audit: PASS]")
    print(f"source_distributions={len(sdists)} members={sdist_members}")
    print(f"wheels={len(wheels)} members={wheel_members}")
    print(f"verified_runtime_files={runtime_files}")
    print(f"frozen_v27_sha256={FROZEN_V27_SHA256}")


if __name__ == "__main__":
    main()
