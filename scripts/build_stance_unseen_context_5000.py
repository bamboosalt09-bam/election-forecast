"""Build a disjoint through-2022 context sample for confirmatory audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.context_corpus import (  # noqa: E402
    ELECTION_CUTOFFS,
    outcome_like_columns,
)


DEFAULT_SOURCE_ROOT = Path(
    r"C:\english_folder\poll_project_post2025_outcome_aware_20260714\outputs\assembly_stance"
)
DEFAULT_OUTPUT = ROOT / "data" / "shadow" / "stance_context_unseen_5000_through2022.csv"
ELECTIONS = tuple(ELECTION_CUTOFFS)
TARGET_TYPES = {"person", "party", "government"}
SEED = "stance-unseen-context-confirmatory-v1"


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def _used_hashes() -> set[str]:
    used: set[str] = set()
    for path in (ROOT / "data" / "shadow").glob("*.csv"):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["text_sha256"])
        except (ValueError, KeyError):
            continue
        used.update(frame["text_sha256"].dropna().astype(str))
    return used


def _validate_source(frame: pd.DataFrame, election_id: str) -> None:
    if set(frame["election_id"].astype(str)) != {election_id}:
        raise ValueError(f"source is not isolated to {election_id}")
    outcome_columns = outcome_like_columns(frame.columns)
    if outcome_columns:
        raise ValueError(f"source contains outcome-like columns: {outcome_columns}")
    dates = pd.to_datetime(frame["meeting_date"], errors="coerce")
    if dates.isna().any() or (dates > ELECTION_CUTOFFS[election_id]).any():
        raise ValueError(f"source violates the {election_id} point-in-time cutoff")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    used = _used_hashes()
    pieces: list[pd.DataFrame] = []
    source_states: list[dict[str, object]] = []
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
        frame["unseen_rank"] = frame["text_sha256"].astype(str).map(_rank)
        polarity = pd.to_numeric(frame["rule_stance_polarity"], errors="coerce").fillna(0)
        directional = frame.loc[polarity.ne(0)].sort_values("unseen_rank")
        neutral = frame.loc[polarity.eq(0)].sort_values("unseen_rank")
        selected = pd.concat(
            [directional, neutral.head(max(1_000 - len(directional), 0))],
            ignore_index=True,
        ).sort_values("unseen_rank").head(1_000)
        if len(selected) != 1_000:
            raise ValueError(f"{election_id} has only {len(selected)} eligible unseen rows")
        selected["context_before"] = ""
        selected["context_after"] = ""
        selected["context_gap_before"] = ""
        selected["context_gap_after"] = ""
        selected["context_current_found"] = 1
        selected["context_group_sentence_count"] = 1
        pieces.append(selected)
        source_states.append(
            {
                "election_id": election_id,
                "source": str(path),
                "eligible_rows": len(frame),
                "selected_rows": len(selected),
                "selected_rule_directional": int(
                    pd.to_numeric(
                        selected["rule_stance_polarity"], errors="coerce"
                    ).fillna(0).ne(0).sum()
                ),
                "latest_meeting_date": pd.to_datetime(
                    selected["meeting_date"]
                ).max().date().isoformat(),
            }
        )
    output = pd.concat(pieces, ignore_index=True)
    if len(output) != 5_000 or output["text_sha256"].duplicated().any():
        raise ValueError("unseen sample must contain 5,000 unique rows")
    if output["text_sha256"].astype(str).isin(used).any():
        raise ValueError("unseen sample overlaps prior training or audits")
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    state = {
        "status": "locked_unseen_through2022_sample",
        "rows": len(output),
        "unique_text_hashes": int(output["text_sha256"].nunique()),
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "context_note": "current sentence only; neighboring context intentionally absent",
        "excluded_prior_hashes": len(used),
        "sources": source_states,
        "output": str(output_path),
    }
    output_path.with_suffix(".state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
