"""Normalize a National Assembly Library export under strict PIT checks.

The National Assembly Library exposes data sources, not a project-specific
stance label. This importer intentionally accepts a small normalized CSV
contract instead of guessing a changing remote API schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.context_corpus import validate_context_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input.resolve(), encoding="utf-8-sig")
    validation = validate_context_corpus(source)
    output = source.copy()
    output["available_date"] = pd.to_datetime(
        output["available_date"]
    ).dt.date.astype(str)
    output["corpus_source"] = output.get(
        "corpus_source", "national_assembly_library"
    )
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    state = {
        "status": "validated_shadow_corpus",
        "active_forecast_changed": False,
        "input": str(args.input.resolve()),
        "output": str(destination),
        "validation": validation.__dict__,
    }
    destination.with_suffix(".state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
