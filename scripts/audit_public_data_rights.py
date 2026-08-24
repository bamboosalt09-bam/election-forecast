"""Fail closed when the public/package data boundary loses source evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/PUBLIC_DATA_SOURCES.json"
# poster_figures/ is audited because the maps are drawn over downloaded SGIS
# administrative boundaries and carry a CC BY 4.0 / KOGL attribution line in the
# image itself. It was outside this list, so the registry could declare coverage
# of poster_figures/v27_ while the published map was v29_ and nothing failed.
AUDITED_ROOTS = (
    "data/",
    "presidential_issue_engine/fixed_dataset/",
    "presidential_issue_engine/poster_figures/",
)
# Files whose content was produced with an external pretrained model. These
# cannot sit inside a family whose basis is this project's own authorship: the
# candidate-issue aggregate was classified under project_authored_and_derived_tables
# and inherited an Apache-2.0 basis that describes project code, not an
# external-model-derived artefact. Each must be covered by a family that names
# the model and states a basis for distributing the derivative.
EXTERNAL_MODEL_DERIVED_PATHS = (
    "data/raw/auto_issue_seed/candidate_issue_profile.csv",
)
EXTERNAL_MODEL_BASIS_TERMS = ("model", "weight")
ALLOWED_DECISIONS = {
    "include-derived-no-source-text",
    "include-derived-with-attribution",
    "include-generated-with-attribution",
}


def publication_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    if register.get("schema") != "election_forecast_public_data_sources_v1":
        raise RuntimeError("public-data register schema drift")
    candidates = publication_files()
    violations: list[str] = []

    for excluded in register["excluded_paths"]:
        if excluded.endswith("/"):
            if any(path.startswith(excluded) for path in candidates):
                violations.append(f"excluded data prefix is tracked: {excluded}")
        elif excluded in candidates:
            violations.append(f"excluded data file is tracked: {excluded}")

    coverage: list[str] = []
    for source in register["sources"]:
        if source.get("redistribution_decision") not in ALLOWED_DECISIONS:
            violations.append(f"unapproved redistribution decision: {source.get('id')}")
        if not source.get("source_url") or not source.get("terms_url"):
            violations.append(f"source or terms URL missing: {source.get('id')}")
        coverage.extend(source.get("coverage_prefixes", []))

    for path in candidates:
        if path.startswith(AUDITED_ROOTS) and not any(path.startswith(prefix) for prefix in coverage):
            violations.append(f"tracked data lacks public-source coverage: {path}")

    for path in EXTERNAL_MODEL_DERIVED_PATHS:
        if path not in candidates:
            continue
        owners = [
            source
            for source in register["sources"]
            if any(path.startswith(prefix) for prefix in source.get("coverage_prefixes", []))
        ]
        if not owners:
            violations.append(f"external-model-derived file is unregistered: {path}")
            continue
        # The file also sits inside a broad directory family, so the most
        # specific covering prefix decides which basis applies.
        def _specificity(source: dict[str, object]) -> int:
            return max(
                (len(prefix) for prefix in source.get("coverage_prefixes", [])
                 if path.startswith(prefix)),
                default=0,
            )

        owner = max(owners, key=_specificity)
        basis = str(owner.get("license_or_basis", "")).casefold()
        if not all(term in basis for term in EXTERNAL_MODEL_BASIS_TERMS):
            violations.append(
                "external-model-derived file lacks a model-aware basis: "
                f"{path} ({owner.get('id')})"
            )

    if violations:
        raise RuntimeError("\n".join(violations))
    print("[public data rights audit: PASS]")
    print(
        "publication_data_files="
        f"{sum(path.startswith(AUDITED_ROOTS) for path in candidates)}"
    )
    print(f"registered_source_families={len(register['sources'])}")
    print("raw_caches_and_uncertain_rights_exports=excluded")


if __name__ == "__main__":
    main()
