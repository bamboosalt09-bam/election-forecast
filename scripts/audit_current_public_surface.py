"""Reject stale public pointers and package-version reuse on current main."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CURRENT_POINTER = ROOT / "data/config/current_presidential_model.json"
ACTIVE_ALIAS = ROOT / "data/config/active_presidential_model.json"
LEGACY_BASE = ROOT / "data/config/active_presidential_model_v16.json"
RELEASE_VERSION = "0.29.0"
MAIN_VERSION = "0.30.0.dev0"
V28_SHA256 = "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba"
V30_SHA256 = "afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e"
V28_PREDICTIONS = ROOT / "outputs/active_presidential_nested_v28/nested_predictions.csv"


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
    require(current.get("active_version") == "v30", "current pointer is not V30")
    require(
        current.get("prediction_sha256") == V30_SHA256,
        "current pointer does not preserve the frozen V30 prediction hash",
    )
    require(
        hashlib.sha256(V28_PREDICTIONS.read_bytes()).hexdigest() == V28_SHA256,
        "frozen V28 rollback prediction hash drifted",
    )
    require(
        current.get("runner") == "scripts/run_active_presidential_model_v30.py",
        "current pointer runner is not V30",
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
    require("run_active_presidential_model_v30 import main" in current_source, "current runner is not V30")

    print("[current public surface audit: PASS]")
    print(f"active_version={current['active_version']}")
    print(f"v30_prediction_sha256={V30_SHA256}")
    print(f"v28_rollback_sha256={V28_SHA256}")
    print(f"package_version={package_version}")
    print(f"frozen_release_version={RELEASE_VERSION}")


if __name__ == "__main__":
    main()
