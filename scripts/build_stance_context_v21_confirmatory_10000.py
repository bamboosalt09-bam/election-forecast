"""Build a fresh 10,000-row confirmatory corpus after V21 is frozen."""

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
TARGET_CUE_RICH = 1_600
ROWS_PER_ELECTION = 2_000
POOL_CAPACITY = {"cue_rich": 1_600, "general": 2_000}
SEED = "stance-context-v21-confirmatory-10000-v1"
_DIRECTIONAL_CUE = re.compile(
    r"실패|무능|부패|비리|잘못|지지|찬성|환영|반대|비판|규탄|"
    r"책임|불신|신뢰|유능|유감|사과|성과|입선|일류|위법|불법"
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


def select_rows(source: Path, excluded: set[str]) -> pd.DataFrame:
    heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, object]]]] = {
        (election, bucket): [] for election in ELECTIONS for bucket in POOL_CAPACITY
    }
    seen = set(excluded)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 500_000 == 0:
                print(f"[V21 confirmatory selection] {row_number:,} rows", flush=True)
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
            key = (election, bucket)
            rank = _rank(text_hash, f"{election}:{bucket}")
            item = (-rank, text_hash, _selected_row(row))
            heap = heaps[key]
            quota = POOL_CAPACITY[bucket]
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    records: list[dict[str, object]] = []
    for election in ELECTIONS:
        cue_heap = heaps[(election, "cue_rich")]
        general_heap = heaps[(election, "general")]
        cue_quota = min(TARGET_CUE_RICH, len(cue_heap))
        general_quota = ROWS_PER_ELECTION - cue_quota
        if len(general_heap) < general_quota:
            raise RuntimeError(
                f"insufficient general rows for {election}: {len(general_heap)} < {general_quota}"
            )
        cue_rows = sorted(cue_heap, key=lambda item: (-item[0], item[1]))[:cue_quota]
        general_rows = sorted(general_heap, key=lambda item: (-item[0], item[1]))[:general_quota]
        records.extend(dict(item[2], selection_bucket="cue_rich") for item in cue_rows)
        records.extend(dict(item[2], selection_bucket="general") for item in general_rows)
    output = pd.DataFrame(records).sort_values(["election_id", "text_sha256"]).reset_index(drop=True)
    if len(output) != 10_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("V21 confirmatory sample is not 10,000 unique rows")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-v21-sha256", required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_rows.csv"
    output_path = output_dir / "stance_context_v21_confirmatory_10000.csv"
    if output_path.exists():
        raise FileExistsError(output_path)
    if selected_path.exists():
        selected = pd.read_csv(selected_path, encoding="utf-8-sig", low_memory=False).fillna("")
        print("[V21 confirmatory selection] resuming", flush=True)
    else:
        excluded = _existing_shadow_hashes()
        print(f"[V21 confirmatory selection] excluded hashes: {len(excluded):,}", flush=True)
        selected = select_rows(source, excluded)
        _atomic_csv(selected, selected_path)
    groups = collect_context(selected, input_path=source)
    output = attach_context(selected, groups)
    _atomic_csv(output, output_path)
    state = {
        "status": "v21_confirmatory_corpus_complete",
        "rows": len(output),
        "seed": SEED,
        "frozen_v21_sha256": args.frozen_v21_sha256,
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
