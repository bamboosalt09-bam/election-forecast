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
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data/config/current_presidential_model.json"
ALIAS = ROOT / "data/config/active_presidential_model.json"

problems: list[str] = []


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
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    version = str(pointer["active_version"])            # e.g. "v29"
    check(re.fullmatch(r"v\d+", version) is not None, f"unexpected active_version: {version}")

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
