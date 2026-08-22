from __future__ import annotations

import pytest

from scripts import audit_distribution_artifacts as audit


def test_sdist_root_is_removed_before_research_boundary_check() -> None:
    with pytest.raises(RuntimeError, match="research-only artifact"):
        audit._assert_public(
            "election_forecast-0.27.0/outputs/v24_defect_ablation/decision.json",
            strip_distribution_root=True,
        )


def test_distribution_member_path_traversal_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="unsafe distribution member"):
        audit._safe_member("../outside.txt")
