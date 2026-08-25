"""Build hooks for the self-contained V30 runtime bundle."""

from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist


ROOT = Path(__file__).resolve().parent
RUNTIME_ARCHIVE = "_v31_runtime.zip"
FIXED_TIMESTAMP = (2026, 8, 22, 0, 0, 0)
V31_ENTRY_MODULES = (
    "scripts.run_current_presidential_model",
    "scripts.run_active_presidential_model_v31",
    "scripts.run_prospective_forecast_v31",
    "scripts.audit_public_active_presidential_model_v31",
    "scripts.verify_v31_clean_reproduction",
    # the intervals builder is a published entry point in its own right;
    # under V30 it reached the wheel only because something else imported
    # it, which is not a property to rely on
    "scripts.build_active_v31_predictive_intervals",
    "presidential_issue_engine.make_poster_figures",
)
LOCAL_MODULE_PREFIXES = ("scripts", "presidential_issue_engine", "common")
RUNTIME_OUTPUT_PREFIXES = (
    "outputs/active_presidential_nested_v31/",
    "outputs/automatic_controls_v22/",
    "outputs/automatic_controls_v23/",
    "outputs/automatic_controls_v26/",
    "outputs/footprint_candidate_base_v9/",
    "outputs/preliminary_slot_assignment/",
    "outputs/preliminary_slot_assignment_v23/",
    "outputs/prospective_pres_2025_v31/",
    "outputs/unified_exact_lineage_v21/",
)
RUNTIME_ROLLBACK_FILES = {
    f"outputs/active_presidential_nested_v{version}/nested_predictions.csv"
    for version in (23, 24, 25, 26, 27, 28, 29, 30)
}
SDIST_PUBLICATION_FILES = {
    ".github/dependabot.yml",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "STRUCTURE.md",
    "docs/AI_MODEL_SPEC.md",
    "docs/ARCHITECTURE.md",
    "docs/COMPETITION_COMPLIANCE_2026.md",
    "docs/DATA_PROVENANCE_AND_REDISTRIBUTION.md",
    "docs/FINAL_MODEL_V29_20260823.md",
    "docs/FINAL_MODEL_V30_20260824.md",
    "docs/FINAL_MODEL_V31_20260825.md",
    "docs/EXPERIMENT_REMOVE_EXTERNAL_MODEL_OVERLAY_20260822.md",
    "docs/EXPERIMENT_V29_THIRD_SHARE_DISPERSION_20260823.md",
    "docs/EXPERIMENT_V30_FORECAST_TIME_WEIGHTS_20260824.md",
    "docs/DIAGNOSIS_SCORING_SCOPE_20260824.md",
    "docs/EXPERIMENT_V31_MULTIPLICATIVE_EXPANSION_20260825.md",
    "docs/METRIC_WEIGHTING_20260825.md",
    "docs/GITHUB_BASELINE_V27_20260822.json",
    "docs/GITHUB_BASELINE_V28_20260823.json",
    "docs/GITHUB_BASELINE_V29_20260823.json",
    "docs/GITHUB_BASELINE_V30_20260824.json",
    "docs/GITHUB_BASELINE_V31_20260825.json",
    "docs/PUBLIC_DATA_SOURCES.json",
    "docs/REPRODUCIBILITY.md",
    "docs/REPOSITORY_BOUNDARIES.md",
    "docs/SBOM.md",
    "docs/VISUALIZATION_DATA.md",
    "pyproject.toml",
    "requirements-v27.lock",
    "requirements-v31.lock",
    "scripts/audit_distribution_artifacts.py",
    "scripts/audit_current_public_surface.py",
    "scripts/audit_github_baseline.py",
    "scripts/audit_public_data_rights.py",
    "scripts/audit_publication_security.py",
    "setup.py",
}
PUBLIC_EXCLUDED_FILES = {
    "presidential_issue_engine/fixed_dataset/kospi_daily.csv",
    "data/raw/kospi_history_source.txt",
    "data/raw/official_sources/assembly_pres_2025_minutes/assembly_stance_rows_2025_h1.csv",
    "data/raw/assembly_issue_character_overlay.csv",
    "data/raw/auto_issue_seed/mega_issue_axis.csv",
    "data/raw/auto_issue_seed/mega_issue_attribution.csv",
    "src/election_forecast/stance_context_model.py",
    "src/election_forecast/stance_context_v15.py",
    "src/election_forecast/stance_context_v16.py",
    "src/election_forecast/stance_context_v17.py",
    "src/election_forecast/stance_context_v18.py",
    "src/election_forecast/stance_context_v19.py",
    "src/election_forecast/stance_context_v20.py",
    "src/election_forecast/stance_context_v21.py",
    "src/election_forecast/stance_context_v22.py",
    "src/election_forecast/stance_context_v23s.py",
    "src/election_forecast/stance_context_v24s.py",
    "src/election_forecast/stance_context_v25s.py",
    "src/election_forecast/stance_explanatory_overlay.py",
    "src/election_forecast/stance_intensity.py",
    "src/election_forecast/stance_precision.py",
    "src/election_forecast/stance_target_policy.py",
    "src/election_forecast/stance_text_model.py",
    "src/election_forecast/stance_v3.py",
}
PUBLIC_EXCLUDED_PREFIXES = (
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


def _is_public_source(relative: str) -> bool:
    relative = relative.replace("\\", "/")
    return (
        relative not in PUBLIC_EXCLUDED_FILES
        and not relative.startswith(PUBLIC_EXCLUDED_PREFIXES)
        and "/__pycache__/" not in f"/{relative}"
        and not relative.endswith((".pyc", ".pyo"))
    )


def _source_files() -> list[str]:
    """Return source-distribution files without ever admitting ignored files."""

    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return sorted(
            item.decode("utf-8").replace("\\", "/")
            for item in result.stdout.split(b"\0")
            if item and _is_public_source(item.decode("utf-8"))
        )

    # An sdist contains only files admitted by MANIFEST.in. Scanning that
    # bounded tree is therefore safe and keeps wheel-from-sdist supported.
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and _is_public_source(path.relative_to(ROOT).as_posix())
        and not any(part in {"build", "dist", "__pycache__", ".egg-info"} for part in path.parts)
    )


def _module_file(module: str) -> Path | None:
    path = ROOT.joinpath(*module.split("."))
    candidate = path.with_suffix(".py")
    if candidate.is_file():
        return candidate
    package = path / "__init__.py"
    return package if package.is_file() else None


def _resolve_relative(module: str, imported: str | None, level: int) -> str:
    parts = module.split(".")[:-1]
    keep = max(0, len(parts) - level + 1)
    prefix = parts[:keep]
    if imported:
        prefix.extend(imported.split("."))
    return ".".join(prefix)


def _python_dependency_closure() -> set[str]:
    """Trace repository-local imports from the public V31 entry points."""

    pending = list(V31_ENTRY_MODULES)
    visited: set[str] = set()
    files: set[str] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = _module_file(module)
        if path is None:
            continue
        files.add(path.relative_to(ROOT).as_posix())
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(module, node.module, node.level)
                    if node.level
                    else (node.module or "")
                )
                if base:
                    candidates.append(base)
                    candidates.extend(f"{base}.{alias.name}" for alias in node.names)
            for candidate in candidates:
                if candidate.startswith(tuple(f"{prefix}." for prefix in LOCAL_MODULE_PREFIXES)):
                    if _module_file(candidate) is not None:
                        pending.append(candidate)
                elif "." not in candidate:
                    # Several historical builders intentionally add both local
                    # source directories to sys.path and then use bare imports.
                    # Resolve those imports as repository modules as well.
                    for prefix in LOCAL_MODULE_PREFIXES:
                        local = f"{prefix}.{candidate}"
                        if _module_file(local) is not None:
                            pending.append(local)
    return files


def v31_runtime_files() -> list[str]:
    """Select the complete, public V30 runtime from admitted source files."""

    finalization = json.loads(
        (ROOT / "outputs/active_presidential_nested_v30/finalization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_artifacts = {
        record["path"]
        for record in finalization["artifacts"]
        if str(record["path"]).startswith(("docs/", "scripts/", "presidential_issue_engine/"))
    }
    runtime_python = _python_dependency_closure()
    included: list[str] = []
    for relative in _source_files():
        if relative in {"LICENSE", "NOTICE"}:
            included.append(relative)
        elif relative in runtime_python:
            included.append(relative)
        elif relative.startswith("presidential_issue_engine/fixed_dataset/"):
            included.append(relative)
        elif relative.startswith(
            (
                "presidential_issue_engine/report/tables/v24/",
                "data/",
            )
        ):
            included.append(relative)
        elif relative == "presidential_issue_engine/report/through2022_rederived/nested_outer_results.csv":
            included.append(relative)
        elif relative.startswith(RUNTIME_OUTPUT_PREFIXES):
            included.append(relative)
        elif relative in RUNTIME_ROLLBACK_FILES:
            included.append(relative)
        elif relative in frozen_artifacts:
            included.append(relative)

    violations = [
        path
        for path in included
        if not _is_public_source(path)
    ]
    if violations:
        raise RuntimeError(f"non-redistributable runtime inputs selected: {violations}")
    return included


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, FIXED_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build_v31_runtime_archive(destination: Path) -> None:
    """Create a deterministic, hash-indexed archive of the public V30 runtime."""

    records = []
    files = v31_runtime_files()
    for relative in files:
        payload = (ROOT / relative).read_bytes()
        records.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema": "election_forecast_v30_packaged_runtime_v1",
        "active_version": "v31",
        "frozen_prediction_sha256": "969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069",
        "source_boundary": "git-tracked-public-files-only",
        "post_2022_outcomes_used": False,
        "files": records,
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w") as archive:
        archive.writestr(_zip_info("_runtime_manifest.json"), manifest_payload)
        for record in records:
            archive.writestr(_zip_info(record["path"]), (ROOT / record["path"]).read_bytes())


class build_py(_build_py):
    """Add the complete V30 runtime after normal Python modules are built."""

    def run(self) -> None:
        super().run()
        for relative in PUBLIC_EXCLUDED_FILES:
            if relative.startswith("src/election_forecast/"):
                built = Path(self.build_lib) / relative.removeprefix("src/")
                if built.is_file():
                    built.unlink()
        build_v31_runtime_archive(Path(self.build_lib) / "election_forecast" / RUNTIME_ARCHIVE)


class sdist(_sdist):
    """Publish the installable V31 source, not the repository research archive."""

    def get_file_list(self) -> None:
        super().get_file_list()
        sources = set(_source_files())
        admitted = set(v31_runtime_files())
        admitted.update(path for path in sources if path.startswith("src/"))
        admitted.update(SDIST_PUBLICATION_FILES)
        self.filelist.files = [
            path
            for path in self.filelist.files
            if path.replace("\\", "/") in admitted
            or path.replace("\\", "/").startswith("src/election_forecast.egg-info/")
        ]


if __name__ == "__main__":
    setup(cmdclass={"build_py": build_py, "sdist": sdist})
