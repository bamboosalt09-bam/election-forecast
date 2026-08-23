from __future__ import annotations

import json
from pathlib import Path

from scripts import run_active_presidential_model as legacy_base


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_public_active_alias_matches_current_v28_pointer() -> None:
    current = load_json("data/config/current_presidential_model.json")
    alias = load_json("data/config/active_presidential_model.json")
    assert alias == current
    assert current["active_version"] == "v28"
    assert current["runner"] == "scripts/run_active_presidential_model_v28.py"


def test_unversioned_base_module_uses_explicit_v16_config() -> None:
    legacy = load_json("data/config/active_presidential_model_v16.json")
    assert legacy_base.CONFIG_PATH.name == "active_presidential_model_v16.json"
    assert legacy["policy_version"] == "active_strict_nested_v16_regional_identity"
