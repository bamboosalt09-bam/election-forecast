"""Reject stale public pointers and package-version reuse on current main."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CURRENT_POINTER = ROOT / "data/config/current_presidential_model.json"
ACTIVE_ALIAS = ROOT / "data/config/active_presidential_model.json"
LEGACY_BASE = ROOT / "data/config/active_presidential_model_v16.json"
RELEASE_VERSION = "0.27.0"
MAIN_VERSION = "0.28.0.dev0"
V27_SHA256 = "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    current = load_json(CURRENT_POINTER)
    alias = load_json(ACTIVE_ALIAS)
    legacy = load_json(LEGACY_BASE)

    require(alias == current, "public active alias differs from current pointer")
    require(current.get("active_version") == "v28", "current pointer is not V28")
    require(
        current.get("prediction_sha256") == V27_SHA256,
        "current pointer does not preserve the V27-equivalent prediction hash",
    )
    require(
        current.get("runner") == "scripts/run_active_presidential_model_v28.py",
        "current pointer runner is not V28",
    )
    require(
        legacy.get("policy_version") == "active_strict_nested_v16_regional_identity",
        "legacy base is not the frozen V16 policy",
    )

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = str(project["project"]["version"])
    require(package_version == MAIN_VERSION, "main package version drift")
    require(package_version != RELEASE_VERSION, "main reuses the frozen release version")

    package_source = (ROOT / "src/election_forecast/__init__.py").read_text(
        encoding="utf-8"
    )
    cli_source = (ROOT / "src/election_forecast/cli.py").read_text(encoding="utf-8")
    base_source = (ROOT / "scripts/run_active_presidential_model.py").read_text(
        encoding="utf-8"
    )
    current_source = (ROOT / "scripts/run_current_presidential_model.py").read_text(
        encoding="utf-8"
    )
    require(f'__version__ = "{MAIN_VERSION}"' in package_source, "package version mismatch")
    require(f'PACKAGE_VERSION = "{MAIN_VERSION}"' in cli_source, "CLI fallback version mismatch")
    require("active_presidential_model_v16.json" in base_source, "base runner reads public active alias")
    require("run_active_presidential_model_v28 import main" in current_source, "current runner is not V28")

    print("[current public surface audit: PASS]")
    print(f"active_version={current['active_version']}")
    print(f"package_version={package_version}")
    print(f"frozen_release_version={RELEASE_VERSION}")


if __name__ == "__main__":
    main()
