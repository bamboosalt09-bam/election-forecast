"""Fail closed when the public/package data boundary loses source evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs/PUBLIC_DATA_SOURCES.json"
AUDITED_ROOTS = ("data/", "presidential_issue_engine/fixed_dataset/")
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
