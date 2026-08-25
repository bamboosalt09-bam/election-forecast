"""V32's scored panel is V31's, unchanged, and that is the point.

V32 repairs the prospective assembly: the target frame used to receive a zero
for every column it lacked, and two of those families were model-active. The
scored panel never went through that assembly, so nothing about 2002-2022
should move.

Rather than assert that, this runner produces the scored artifact by running
V31's chain into V32's directory and then requires the result to be **byte
identical** to V31's. If a single row differs, something other than the
prospective assembly changed, and the promotion stops here rather than shipping
a scored artifact nobody meant to move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v31 as v31  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v32"
V31_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v31"
FINAL_VARIANT = "v32_prospective_feature_contract"
V31_PREDICTION_SHA256 = "969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)

    produced = v31.run(output_dir=destination)
    predictions = Path(produced) / "nested_predictions.csv"
    digest = _sha256(predictions)
    if digest != V31_PREDICTION_SHA256:
        raise RuntimeError(
            "V32's scored panel differs from V31's. This version changes the "
            "prospective assembly only, so a scored difference means something "
            f"else was mixed in. Expected {V31_PREDICTION_SHA256}, got {digest}."
        )

    summary_path = Path(produced) / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["policy_version"] = "active_v32_prospective_feature_contract"
    payload["predecessor"] = "v31"
    payload["scored_panel_identical_to_v31"] = True
    payload["scored_panel_sha256"] = digest
    payload["prospective_feature_contract"] = {
        "changed": "the prospective target assembly only",
        "scored_panel_effect": "none; byte identical to V31 by construction and by check",
    }
    payload["metrics"]["variant"] = FINAL_VARIANT
    v24._atomic_json_crlf(payload, summary_path)
    return Path(produced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    destination = run(parser.parse_args().output_dir)
    print(v24.report(destination).to_string(index=False))


if __name__ == "__main__":
    main()
