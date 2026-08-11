"""Build a fresh confirmatory corpus after stance V23-S is frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_stance_context_v21_confirmatory_10000 as base  # noqa: E402
from scripts.build_stance_context_5000 import attach_context, collect_context  # noqa: E402
from scripts.build_stance_context_broad_25000 import _existing_shadow_hashes  # noqa: E402


SEED = "stance-context-v23s-confirmatory-10000-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-v23s-sha256", required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_rows.csv"
    output_path = output_dir / "stance_context_v23s_confirmatory_10000.csv"
    if output_path.exists():
        raise FileExistsError(output_path)
    base.SEED = SEED
    if selected_path.exists():
        selected = pd.read_csv(selected_path, encoding="utf-8-sig", low_memory=False).fillna("")
        print("[V23-S confirmatory selection] resuming", flush=True)
    else:
        excluded = _existing_shadow_hashes()
        print(f"[V23-S confirmatory selection] excluded hashes: {len(excluded):,}", flush=True)
        selected = base.select_rows(source, excluded)
        base._atomic_csv(selected, selected_path)
    groups = collect_context(selected, input_path=source)
    output = attach_context(selected, groups)
    base._atomic_csv(output, output_path)
    state = {
        "status": "v23s_confirmatory_corpus_complete",
        "rows": len(output),
        "seed": SEED,
        "frozen_v23s_sha256": args.frozen_v23s_sha256,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "target_counts": output["target_type"].value_counts().to_dict(),
        "selection_bucket_counts": output["selection_bucket"].value_counts().to_dict(),
        "output": str(output_path),
    }
    (output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
