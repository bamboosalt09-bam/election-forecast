"""Fail when any version declaration disagrees with the active-model pointer.

Promoting a model means editing a version token in a dozen unrelated places -
the pointer, the package version, the CLI banner, the packaged runtime loader,
the distribution allowlists, the frozen-hash constants, the CI job names. Every
promotion in this repository has missed at least one of them, and the ways they
surface are uneven: a wheel build fails loudly, a stale audit constant fails
only from an installed wheel, a dead runtime module ships silently, and a
renamed CI job blocks main forever with every check green.

This is the single check that makes all of those the same failure. The pointer
at data/config/current_presidential_model.json is the source of truth; every
other declaration must agree with it.

``--fix`` rewrites the declarations whose correct value is *computable* from the
pointer: the document markers, the package version wherever it is repeated, the
CLI banner and the frozen prediction hash. It deliberately does not touch prose,
rename modules or invent documents - those need judgment, and a tool that
guessed at them would produce confident nonsense. Fixer and checker share one
table, so they cannot drift apart: whatever ``--fix`` writes is exactly what the
check reads back.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import argparse
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data/config/current_presidential_model.json"
ALIAS = ROOT / "data/config/active_presidential_model.json"

problems: list[str] = []

#: Every document that describes the *current* model, as opposed to recording a
#: past one. Each carries a machine-readable marker on its first line, because
#: prose is not a reliable declaration: before the V29 sync these files named
#: V28 as active while already citing v29 script paths, so a token search passed
#: them all.
CORE_PUBLIC_DOCUMENTS = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/COMPETITION_COMPLIANCE_2026.md",
    "docs/HANDOFF_CURRENT_STATE.md",
    "docs/REPOSITORY_BOUNDARIES.md",
    "docs/SBOM.md",
    "docs/AI_MODEL_SPEC.md",
    "docs/VISUALIZATION_DATA.md",
    "docs/GITHUB_WORKFLOW.md",
    "docs/DATA_PROVENANCE_AND_REDISTRIBUTION.md",
    "docs/PRES_2025_INPUT_GUIDE.md",
    "SECURITY.md",
)
#: Versions that may legitimately appear in paths: the active one and its
#: frozen rollbacks, which the audits pin by design.
ROLLBACK_VERSIONS = frozenset({"v23", "v24", "v25", "v26", "v27", "v28"})
ACTIVE_VERSION_MARKER = re.compile(r"<!--\s*active-model-version:\s*(v\d+)\s*-->")
#: The prose assertions are still checked, but only above the first dated or
#: superseded heading. A handoff log legitimately records "active V23" inside a
#: 2026-08-02 section; what must not happen is the *preamble* naming an old
#: version, which is exactly how HANDOFF_CURRENT_STATE went stale.
HISTORICAL_HEADING = re.compile(r"^#{2,}\s.*(?:\d{4}-\d{2}-\d{2}|20\d{6}|[Ss]uperseded)", re.MULTILINE)
ACTIVENESS_PATTERNS = (
    r"[Aa]ctive (V\d+)",
    r"(V\d+) is the active",
    r"활성 (V\d+)",
)


#: Sites whose correct content follows mechanically from the pointer. Each entry
#: is (path, pattern with one capturing group, callable returning the value).
#: The check compares the group; --fix rewrites it. One table, both directions.
def _mechanical_sites(version: str, package_version: str, prediction_hash: str):
    sites = [
        ("pyproject.toml", r'(?m)^version = "([^"]+)"', package_version),
        ("src/election_forecast/__init__.py", r'__version__ = "([^"]+)"', package_version),
        ("src/election_forecast/cli.py", r'PACKAGE_VERSION = "([^"]+)"', package_version),
        ("scripts/audit_current_public_surface.py", r'MAIN_VERSION = "([^"]+)"', package_version),
        ("src/election_forecast/cli.py", r'ACTIVE_MODEL_VERSION = "([^"]+)"', version.upper()),
        ("setup.py", r'"frozen_prediction_sha256": "([0-9a-f]{64})"', prediction_hash),
        ("scripts/audit_distribution_artifacts.py", r'FROZEN_V\d+_SHA256 = \(\s*"([0-9a-f]{64})"', prediction_hash),
        (
            "scripts/audit_current_public_surface.py",
            rf'V{version[1:]}_SHA256 = "([0-9a-f]{{64}})"',
            prediction_hash,
        ),
    ]
    sites += [
        (relative, r"<!--\s*active-model-version:\s*(v\d+)\s*-->", version)
        for relative in CORE_PUBLIC_DOCUMENTS
    ]
    return sites


def apply_fixes(sites) -> list[str]:
    """Rewrite each mechanical site to its computed value; report what changed."""

    changed: list[str] = []
    for relative, pattern, expected in sites:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        found = re.search(pattern, text)
        if found is None or found.group(1) == expected:
            continue
        start, end = found.span(1)
        path.write_bytes((text[:start] + expected + text[end:]).encode("utf-8"))
        changed.append(f"{relative}: {found.group(1)} -> {expected}")
    return changed


def check(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _single(pattern: str, text: str, label: str) -> str | None:
    found = re.search(pattern, text)
    if found is None:
        problems.append(f"could not find {label}")
        return None
    return found.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite the declarations whose value follows from the pointer, then re-check",
    )
    arguments = parser.parse_args()

    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    version = str(pointer["active_version"])            # e.g. "v29"
    check(re.fullmatch(r"v\d+", version) is not None, f"unexpected active_version: {version}")

    if arguments.fix:
        # pyproject is the package version's own source; the pointer supplies
        # the rest. Nothing here is guessed - every value is read, not invented.
        changed = apply_fixes(
            _mechanical_sites(
                version,
                str(tomllib.loads(_read("pyproject.toml"))["project"]["version"]),
                hashlib.sha256(
                    (ROOT / str(pointer["output"]) / "nested_predictions.csv").read_bytes()
                ).hexdigest(),
            )
        )
        for line in changed:
            print(f"  fixed {line}")
        if not changed:
            print("  nothing mechanical to fix")
        print()

    # 1. the alias must be the same document, not a copy that drifted
    check(json.loads(ALIAS.read_text(encoding="utf-8")) == pointer,
          "data/config/active_presidential_model.json differs from the current pointer")

    # 2. every pointer field that names a version must name this one
    for field in ("runner", "prospective_runner", "version_wrapper", "output",
                  "canonical_document", "finalization_manifest"):
        value = str(pointer[field])
        check(version in value.lower() or version.upper() in value,
              f"pointer.{field} does not name {version}: {value}")
        check((ROOT / value).exists(), f"pointer.{field} points at a missing path: {value}")

    # 3. one package version, declared four times
    package_version = str(tomllib.loads(_read("pyproject.toml"))["project"]["version"])
    for relative, pattern, label in (
        ("src/election_forecast/__init__.py", r'__version__ = "([^"]+)"', "__version__"),
        ("src/election_forecast/cli.py", r'PACKAGE_VERSION = "([^"]+)"', "cli PACKAGE_VERSION"),
        ("scripts/audit_current_public_surface.py", r'MAIN_VERSION = "([^"]+)"', "MAIN_VERSION"),
    ):
        declared = _single(pattern, _read(relative), label)
        check(declared == package_version,
              f"{label} is {declared}, pyproject says {package_version}")

    # 4. the CLI must announce the active model, not a predecessor
    cli = _read("src/election_forecast/cli.py")
    banner = _single(r'ACTIVE_MODEL_VERSION = "([^"]+)"', cli, "ACTIVE_MODEL_VERSION")
    check(banner == version.upper(), f"CLI announces {banner}, pointer says {version.upper()}")

    # 5. exactly one packaged runtime loader, and it is this version's.
    #    A predecessor's loader left behind ships a module pointing at an
    #    archive the wheel no longer contains - dead on arrival and invisible.
    loaders = sorted(p.name for p in (ROOT / "src/election_forecast").glob("v*_runtime.py"))
    check(loaders == [f"{version}_runtime.py"],
          f"packaged runtime loaders are {loaders}, expected only {version}_runtime.py")
    check(f"from election_forecast.{version}_runtime import" in cli,
          f"the CLI does not import the {version} runtime loader")

    # 6. the packaged archive name and the setup entry points
    setup = _read("setup.py")
    archive = _single(r'RUNTIME_ARCHIVE = "([^"]+)"', setup, "RUNTIME_ARCHIVE")
    check(archive == f"_{version}_runtime.zip",
          f"setup.py bundles {archive}, expected _{version}_runtime.zip")
    check(f'election_forecast = ["_{version}_runtime.zip"]' in _read("pyproject.toml"),
          f"pyproject does not package _{version}_runtime.zip")
    check(str(pointer["canonical_document"]) in setup,
          "the canonical document is not in the distribution allowlist")

    # 7. the frozen prediction hash, declared in three places
    prediction = ROOT / str(pointer["output"]) / "nested_predictions.csv"
    actual = hashlib.sha256(prediction.read_bytes()).hexdigest()
    check(str(pointer["prediction_sha256"]) == actual,
          "pointer.prediction_sha256 does not match the artifact")
    for relative, pattern, label in (
        ("setup.py", r'"frozen_prediction_sha256": "([0-9a-f]{64})"', "setup frozen_prediction_sha256"),
        ("scripts/audit_distribution_artifacts.py", r'FROZEN_V\d+_SHA256 = \(\s*"([0-9a-f]{64})"', "distribution FROZEN hash"),
        ("scripts/audit_current_public_surface.py", rf'V{version[1:]}_SHA256 = "([0-9a-f]{{64}})"', "surface active hash"),
    ):
        declared = _single(pattern, _read(relative), label)
        check(declared == actual, f"{label} is stale")

    # 8. the GitHub baseline the boundary audit reads must be this version's
    baseline_name = _single(r'BASELINE = ROOT / "docs" / "([^"]+)"',
                            _read("scripts/audit_github_baseline.py"), "BASELINE")
    if baseline_name:
        baseline = json.loads(_read(f"docs/{baseline_name}"))
        check(str(baseline["active_version"]) == version,
              f"docs/{baseline_name} declares {baseline['active_version']}, pointer says {version}")

    # 9. CI job names must not carry a version. Branch protection pins required
    #    checks by name, so a versioned job has to be renamed every promotion
    #    and the protection rule updated in lockstep - and when that is missed,
    #    main blocks on a check that will never report again, with every job
    #    green and nothing to point at.
    # 10. every current-state document must carry the marker for this version,
    #     and any activeness sentence in it must agree. A marker that disagrees
    #     with its own prose is worse than either alone.
    upper = version.upper()
    for relative in CORE_PUBLIC_DOCUMENTS:
        text = _read(relative)
        marked = ACTIVE_VERSION_MARKER.findall(text)
        if not marked:
            problems.append(
                f"{relative} carries no <!-- active-model-version: {version} --> marker"
            )
        elif set(marked) != {version}:
            problems.append(
                f"{relative} is marked {', '.join(sorted(set(marked)))}, pointer says {version}"
            )
        cut = HISTORICAL_HEADING.search(text)
        preamble = text[: cut.start()] if cut else text
        declared = {
            match
            for pattern in ACTIVENESS_PATTERNS
            for match in re.findall(pattern, preamble)
        }
        stale = sorted(declared - {upper})
        if stale:
            problems.append(
                f"{relative} still calls {', '.join(stale)} active in prose"
            )

    # 11. the audits' own required-file lists name versioned paths, and nothing
    #     was checking them. A distribution audit that still demands the
    #     predecessor's prospective runner passes right up until a wheel is
    #     built without it.
    #
    #     Rollback versions are allowed in *artifact* paths - the frozen
    #     prediction CSVs are pinned by design - but not in executable ones. A
    #     first attempt at this check exempted rollbacks everywhere and so let
    #     `scripts/run_prospective_forecast_v28.py` through, which is the exact
    #     defect it exists to catch. Intermediate directories such as
    #     automatic_controls_v22 carry unrelated version numbers and are not
    #     examined at all.
    executable_paths = re.compile(
        r"(?:scripts/(?:run_active_presidential_model|run_prospective_forecast|"
        r"audit_public_active_presidential_model|build_active|"
        r"finalize_active_presidential_model|verify)_?\w*?(v\d+)\w*\.py"
        r"|src/election_forecast/(v\d+)_runtime)"
    )
    artifact_paths = re.compile(
        r"outputs/(?:active_presidential_nested|prospective_pres_2025)_(v\d+)/"
    )
    for relative in ("scripts/audit_distribution_artifacts.py", "setup.py"):
        text = _read(relative)
        executable = {g for match in executable_paths.findall(text) for g in match if g}
        stale = sorted(v for v in executable if v != version)
        check(
            not stale,
            f"{relative} requires the {', '.join(stale)} runner or loader "
            f"where {version} is active",
        )
        artifacts = set(artifact_paths.findall(text))
        unknown = sorted(v for v in artifacts if v != version and v not in ROLLBACK_VERSIONS)
        check(
            not unknown,
            f"{relative} names {', '.join(unknown)} artifact paths that are neither "
            f"active nor a declared rollback",
        )

    # 12. the canonical dependency lock, wherever it is named as the reproduction
    #     environment. Three documents pointed at the superseded V27 lock while
    #     CI audited the V29 one.
    lock = f"requirements-{version}.lock"
    check((ROOT / lock).is_file(), f"the canonical lock {lock} does not exist")
    for relative in ("docs/SBOM.md", "docs/COMPETITION_COMPLIANCE_2026.md",
                     "README.md", "scripts/audit_distribution_artifacts.py"):
        text = _read(relative)
        check(lock in text, f"{relative} never names the canonical lock {lock}")

    workflow = _read(".github/workflows/ci.yml")
    versioned_jobs = re.findall(r"^  ([a-z0-9-]*v\d+[a-z0-9-]*):", workflow, re.MULTILINE)
    check(not versioned_jobs,
          f"CI job names carry a version and will break branch protection: {versioned_jobs}")

    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(f"version consistency: {len(problems)} declaration(s) disagree with the pointer")
    print("[version consistency audit: PASS]")
    print(f"active_version={version}")
    print(f"package_version={package_version}")
    print(f"prediction_sha256={actual}")


if __name__ == "__main__":
    main()
