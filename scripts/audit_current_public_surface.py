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
RELEASE_VERSION = "0.30.0"
MAIN_VERSION = "0.31.0.dev0"
V31_SHA256 = "969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069"
#: Every frozen predecessor, not just one. This pinned V28 alone and kept
#: doing so after V29 was promoted and then rolled back under V30, so the
#: newest rollback - the one a bad promotion is most likely to disturb -
#: was the one nothing here checked.
ROLLBACK_SHA256 = {
    "v23": "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b",
    "v24": "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a",
    "v25": "218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b",
    "v26": "9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3",
    "v27": "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b",
    "v28": "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba",
    "v29": "fed959cdba1e127f91c2ab640a378d1f44a4a3e79b4c4a76893cf8d7c6153904",
    "v30": "afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e",
}


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
    require(current.get("active_version") == "v31", "current pointer is not V31")
    require(
        current.get("prediction_sha256") == V31_SHA256,
        "current pointer does not preserve the frozen V30 prediction hash",
    )
    for version, expected in sorted(ROLLBACK_SHA256.items()):
        predictions = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        require(predictions.is_file(), f"frozen {version.upper()} rollback prediction is missing")
        require(
            hashlib.sha256(predictions.read_bytes()).hexdigest() == expected,
            f"frozen {version.upper()} rollback prediction hash drifted",
        )
    require(
        str(current.get("predecessor")) in ROLLBACK_SHA256,
        "the pointer's predecessor is not a pinned rollback",
    )
    require(
        current.get("runner") == "scripts/run_active_presidential_model_v31.py",
        "current pointer runner is not V31",
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
    require("run_active_presidential_model_v31 import main" in current_source, "current runner is not V31")

    print("[current public surface audit: PASS]")
    print(f"active_version={current['active_version']}")
    print(f"v31_prediction_sha256={V31_SHA256}")
    print(f"pinned_rollbacks={','.join(sorted(ROLLBACK_SHA256))}")
    print(f"predecessor={current['predecessor']}")
    print(f"package_version={package_version}")
    print(f"frozen_release_version={RELEASE_VERSION}")


if __name__ == "__main__":
    main()
