"""Rebuild V27 outside its frozen directory and verify exact predictions."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_active_presidential_model_v27  # noqa: E402


FROZEN = ROOT / "outputs/active_presidential_nested_v27/nested_predictions.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="election_forecast_v27_") as temporary:
        destination = Path(temporary) / "active_presidential_nested_v27"
        run_active_presidential_model_v27.run(destination)
        reproduced = destination / "nested_predictions.csv"
        if sha256(reproduced) != sha256(FROZEN):
            raise RuntimeError(
                "clean V27 reproduction differs from the frozen prediction artifact: "
                f"{sha256(reproduced)} != {sha256(FROZEN)}"
            )
        print("[clean V27 reproduction: PASS]")
        print(f"prediction_sha256={sha256(reproduced)}")


if __name__ == "__main__":
    main()
