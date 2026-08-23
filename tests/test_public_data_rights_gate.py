"""Guards for the external-model-derived registration gate.

`data/raw/auto_issue_seed/candidate_issue_profile.csv` was covered only by
`project_authored_and_derived_tables`, whose basis is Apache-2.0 project
authorship. That describes the transformation code, not an artefact produced
with an external pretrained encoder, so the rights register asserted a basis it
did not have. These tests pin the separate registration and the gate that keeps
it from silently reverting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_public_data_rights as rights

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = "data/raw/auto_issue_seed/candidate_issue_profile.csv"


def _register() -> dict:
    return json.loads(rights.REGISTER.read_text(encoding="utf-8"))


def test_the_aggregate_is_named_as_external_model_derived() -> None:
    assert AGGREGATE in rights.EXTERNAL_MODEL_DERIVED_PATHS


def test_it_has_a_family_whose_basis_names_the_model() -> None:
    register = _register()
    owners = [
        source
        for source in register["sources"]
        if any(AGGREGATE.startswith(prefix) for prefix in source["coverage_prefixes"])
    ]
    assert owners, "the aggregate must be covered by some family"

    def specificity(source: dict) -> int:
        return max(
            len(prefix)
            for prefix in source["coverage_prefixes"]
            if AGGREGATE.startswith(prefix)
        )

    owner = max(owners, key=specificity)
    basis = str(owner["license_or_basis"]).casefold()
    assert all(term in basis for term in rights.EXTERNAL_MODEL_BASIS_TERMS)
    # the basis must not silently claim a grant the model card never gave
    assert "no explicit license tag" in basis or "no redistribution grant" in basis


def test_the_project_authorship_family_no_longer_owns_it_alone() -> None:
    """The specific family must out-rank the broad directory family."""

    register = _register()
    by_id = {source["id"]: source for source in register["sources"]}
    specific = by_id.get("external_model_derived_candidate_issue_aggregate")
    assert specific is not None
    broad = by_id["project_authored_and_derived_tables"]

    def longest(source: dict) -> int:
        return max(
            (len(prefix) for prefix in source["coverage_prefixes"] if AGGREGATE.startswith(prefix)),
            default=0,
        )

    assert longest(specific) > longest(broad)


def test_the_declared_basis_states_what_the_file_does_not_contain() -> None:
    """A derivative that embedded weights or source text would need a grant."""

    register = _register()
    by_id = {source["id"]: source for source in register["sources"]}
    basis = str(by_id["external_model_derived_candidate_issue_aggregate"]["license_or_basis"])
    lowered = basis.casefold()
    assert "no model weight" in lowered
    assert "no source sentence" in lowered


def test_the_audit_passes_on_the_current_register() -> None:
    rights.main()


def test_the_gate_rejects_a_basis_that_does_not_mention_a_model() -> None:
    """Reverting to the project-authorship classification must fail the audit."""

    assert rights.EXTERNAL_MODEL_BASIS_TERMS, "an empty term set would pass anything"
    apache_basis = (
        "Apache-2.0 for project-authored schemas, annotations and transformations"
    ).casefold()
    assert not all(term in apache_basis for term in rights.EXTERNAL_MODEL_BASIS_TERMS)
