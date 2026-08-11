"""Build a fresh cue-targeted V23-S holdout from the frozen full corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import re
import sys
import uuid
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stance_context_5000 import attach_context, collect_context  # noqa: E402
from scripts.build_stance_context_broad_25000 import _existing_shadow_hashes  # noqa: E402


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
TARGET_TYPES = {"person", "party", "government"}
ROWS_PER_ELECTION = 1_000
SEED = "stance-context-v23s-targeted-holdout-5000-v1"
_DIRECTIONAL_CUE = re.compile(
    r"실패|무능|부패|비리|잘못|지지|찬성|환영|반대|비판|규탄|"
    r"책임|불신|신뢰|유능|유감|성과|성공|낙선|일류|위법|불법|"
    r"우려|심각|문제|개선|사퇴|퇴진|탄핵|거짓|왜곡|반성|사과|"
    r"칭찬|훌륭|존경|실망|분노|개탄|방조|특혜|의혹|파탄|실정"
)


def _rank(text_hash: str, bucket: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{bucket}|{text_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _selected_row(row: dict[str, str]) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    output["rule_stance_label"] = row.get("stance_label", "")
    output["rule_stance_polarity"] = row.get("stance_polarity", "0")
    output["rule_stance_confidence"] = row.get("stance_confidence", "")
    output["rule_stance_cues"] = row.get("stance_cues", "")
    return output


def _push(
    heap: list[tuple[int, str, dict[str, object]]],
    row: dict[str, str],
    election: str,
    bucket: str,
) -> None:
    text_hash = str(row["text_sha256"])
    rank = _rank(text_hash, f"{election}:{bucket}")
    item = (-rank, text_hash, _selected_row(row))
    if len(heap) < ROWS_PER_ELECTION:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def select_rows(source: Path, excluded: set[str]) -> pd.DataFrame:
    heaps = {
        (election, bucket): []
        for election in ELECTIONS
        for bucket in ("cue_rich", "general")
    }
    seen = set(excluded)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 500_000 == 0:
                print(f"[V23-S targeted selection] {row_number:,} rows", flush=True)
            election = str(row.get("election_id", ""))
            target_type = str(row.get("target_type", ""))
            text_hash = str(row.get("text_sha256", ""))
            if (
                election not in ELECTIONS
                or target_type not in TARGET_TYPES
                or not text_hash
                or text_hash in seen
            ):
                continue
            seen.add(text_hash)
            bucket = (
                "cue_rich"
                if _DIRECTIONAL_CUE.search(str(row.get("text_excerpt", "")))
                else "general"
            )
            _push(heaps[(election, bucket)], row, election, bucket)

    records: list[dict[str, object]] = []
    for election in ELECTIONS:
        cue = sorted(heaps[(election, "cue_rich")], key=lambda item: (-item[0], item[1]))
        general = sorted(heaps[(election, "general")], key=lambda item: (-item[0], item[1]))
        cue_quota = min(ROWS_PER_ELECTION, len(cue))
        general_quota = ROWS_PER_ELECTION - cue_quota
        if len(general) < general_quota:
            raise RuntimeError(f"Insufficient untouched rows for {election}")
        records.extend(dict(item[2], selection_bucket="cue_rich") for item in cue[:cue_quota])
        records.extend(
            dict(item[2], selection_bucket="general") for item in general[:general_quota]
        )
    output = pd.DataFrame(records).sort_values(["election_id", "text_sha256"]).reset_index(drop=True)
    expected_rows = ROWS_PER_ELECTION * len(ELECTIONS)
    if len(output) != expected_rows or output["text_sha256"].duplicated().any():
        raise RuntimeError(
            f"Targeted V23-S holdout is not {expected_rows:,} unique rows"
        )
    if output["text_sha256"].astype(str).isin(excluded).any():
        raise RuntimeError("Targeted V23-S holdout overlaps prior shadow data")
    return output


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
    output_path = output_dir / "stance_context_v23s_targeted_holdout_5000.csv"
    if output_path.exists():
        raise FileExistsError(output_path)

    excluded = _existing_shadow_hashes()
    print(f"[V23-S targeted selection] excluded hashes: {len(excluded):,}", flush=True)
    selected = select_rows(source, excluded)
    _atomic_csv(selected, selected_path)
    groups = collect_context(selected, input_path=source)
    output = attach_context(selected, groups)
    _atomic_csv(output, output_path)
    state = {
        "status": "v23s_targeted_holdout_complete",
        "rows": len(output),
        "seed": SEED,
        "frozen_v23s_sha256": args.frozen_v23s_sha256,
        "content_reviewed_before_selection": False,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "selection_bucket_counts": output["selection_bucket"].value_counts().to_dict(),
        "output": str(output_path),
    }
    (output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
