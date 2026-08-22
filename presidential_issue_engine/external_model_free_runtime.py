"""Runtime guard that excludes external neural inference and direct overlays.

V28 keeps the frozen historical candidate-issue aggregate required by the
validated postprocess, but it does not execute a neural encoder or read the
sentence-level stance overlay.  The environment guard is process-wide so it
also covers historical modules imported under a bare name during the legacy
execution chain.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Iterator

import pandas as pd


OVERLAY_ENV = "POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"
OVERLAY_PATH_FRAGMENT = "assembly_issue_character_overlay.csv"
DERIVED_SEED_FRAGMENTS = (
    "data/raw/auto_issue_seed/mega_issue_axis.csv",
    "data/raw/auto_issue_seed/mega_issue_attribution.csv",
)


def _engine_modules() -> list[object]:
    return [
        module
        for name, module in tuple(sys.modules.items())
        if name.endswith("issue_vote_engine")
        and hasattr(module, "THROUGH_2022_REDERIVED_LAYER_CONFIG")
    ]


@contextmanager
def external_model_free_runtime() -> Iterator[None]:
    """Disable neural runtime consumers and restore caller state on exit."""

    previous = os.environ.get(OVERLAY_ENV)
    patched: list[tuple[object, dict[str, object], dict[str, object]]] = []
    os.environ[OVERLAY_ENV] = "disabled"
    for module in _engine_modules():
        config = module.THROUGH_2022_REDERIVED_LAYER_CONFIG
        registry = module.THROUGH_2022_LAYER_REGISTRY
        patched.append((module, dict(config), dict(registry)))
        config["overlay_gain"] = 0.0
        config["automatic_issue_seed_enabled"] = False
        config["manual_issue_seed_enabled"] = False
        registry["issue_character_overlay"] = {"enabled": False}
        registry["automatic_issue_seed"] = {"enabled": False}
    try:
        yield
    finally:
        for module, config, registry in patched:
            module.THROUGH_2022_REDERIVED_LAYER_CONFIG.clear()
            module.THROUGH_2022_REDERIVED_LAYER_CONFIG.update(config)
            module.THROUGH_2022_LAYER_REGISTRY.clear()
            module.THROUGH_2022_LAYER_REGISTRY.update(registry)
        if previous is None:
            os.environ.pop(OVERLAY_ENV, None)
        else:
            os.environ[OVERLAY_ENV] = previous


def strip_external_model_inputs(manifest_path: Path) -> None:
    """Remove direct overlay inputs while retaining the disclosed frozen profile."""

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    paths = manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    forbidden = paths.str.endswith(OVERLAY_PATH_FRAGMENT)
    for fragment in DERIVED_SEED_FRAGMENTS:
        forbidden |= paths.str.endswith(fragment)
    filtered = manifest.loc[~forbidden].copy()
    filtered.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def assert_external_model_free_manifest(manifest_path: Path) -> None:
    """Fail closed if a direct overlay or unused derived seed remains declared."""

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    paths = manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    forbidden = paths.str.endswith(OVERLAY_PATH_FRAGMENT)
    for fragment in DERIVED_SEED_FRAGMENTS:
        forbidden |= paths.str.endswith(fragment)
    if forbidden.any():
        raise RuntimeError("external-model-derived input remains in manifest")
