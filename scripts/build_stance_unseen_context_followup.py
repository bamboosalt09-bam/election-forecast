"""Build the remaining disjoint through-2022 sample for a second audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_stance_unseen_context_5000 import (
    DEFAULT_SOURCE_ROOT,
    ELECTIONS,
    ROOT,
    TARGET_TYPES,
    _used_hashes,
    _validate_source,
)


DEFAULT_OUTPUT = ROOT / "data" / "shadow" / "stance_context_unseen_followup_4000_through2022.csv"
SEED = "stance-unseen-context-confirmatory-v2"


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=4_000)
    args = parser.parse_args()
    used = _used_hashes()
    pieces: list[pd.DataFrame] = []
    for election_id in ELECTIONS:
        path = (
            args.source_root.resolve()
            / f"pilot_{election_id}_5000"
            / "review_batch.csv"
        )
        frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        _validate_source(frame, election_id)
        frame = frame.loc[
            frame["target_type"].isin(TARGET_TYPES)
            & ~frame["text_sha256"].astype(str).isin(used)
        ].drop_duplicates("text_sha256").copy()
        pieces.append(frame)
    eligible = pd.concat(pieces, ignore_index=True).drop_duplicates("text_sha256")
    if len(eligible) < args.rows:
        raise ValueError(f"requested {args.rows} rows but only {len(eligible)} remain")
    eligible["unseen_rank"] = eligible["text_sha256"].astype(str).map(_rank)
    polarity = pd.to_numeric(
        eligible["rule_stance_polarity"], errors="coerce"
    ).fillna(0)
    selected = pd.concat(
        [
            eligible.loc[polarity.ne(0)].sort_values("unseen_rank"),
            eligible.loc[polarity.eq(0)].sort_values("unseen_rank"),
        ],
        ignore_index=True,
    ).head(args.rows)
    selected["context_before"] = ""
    selected["context_after"] = ""
    selected["context_gap_before"] = ""
    selected["context_gap_after"] = ""
    selected["context_current_found"] = 1
    selected["context_group_sentence_count"] = 1
    if selected["text_sha256"].astype(str).isin(used).any():
        raise ValueError("follow-up sample overlaps earlier development or audits")
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    state = {
        "status": "locked_unseen_followup_through2022_sample",
        "rows": len(selected),
        "eligible_rows_before_selection": len(eligible),
        "unique_text_hashes": int(selected["text_sha256"].nunique()),
        "excluded_prior_hashes": len(used),
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "election_counts": selected["election_id"].value_counts().sort_index().to_dict(),
        "latest_meeting_dates": {
            key: value.date().isoformat()
            for key, value in pd.to_datetime(selected["meeting_date"])
            .groupby(selected["election_id"])
            .max()
            .items()
        },
        "output": str(output_path),
    }
    output_path.with_suffix(".state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
