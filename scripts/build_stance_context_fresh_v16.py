"""Build a fresh, resumable 5,000-row target-aware V16 audit corpus."""

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


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
TARGET_TYPES = {"person", "party", "government"}
QUOTAS = {"cue_rich": 800, "general": 200}
DEFAULT_SEED = "stance-context-fresh-v16-confirmatory"
_DIRECTIONAL_CUE = re.compile(
    r"실패|무능|부패|비리|잘못|지지|찬성|환영|반대|비판|규탄|"
    r"책임|불신|신뢰|유감|훌륭|성과|독선|혼란|위법|불법"
)


def _rank(text_hash: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}|{text_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _excluded_hashes() -> set[str]:
    paths = [
        ROOT / "data" / "shadow" / "stance_context_5000_through2022.csv",
        ROOT / "data" / "shadow" / "stance_context_unseen_5000_through2022.csv",
        ROOT / "data" / "shadow" / "stance_context_unseen_followup_4000_through2022.csv",
        ROOT
        / "data"
        / "shadow"
        / "stance_context_fresh_v16"
        / "stance_context_fresh_v16_5000.csv",
        ROOT
        / "data"
        / "shadow"
        / "stance_context_fresh_v16_supplement"
        / "stance_context_fresh_v16_5000.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_a.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_b.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v9.csv",
        ROOT
        / "data"
        / "shadow"
        / "stance_context_fresh_v18_confirmatory"
        / "stance_context_fresh_v18_5000.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v10_part_a.csv",
        ROOT
        / "data"
        / "shadow"
        / "stance_context_fresh_v18_supplement"
        / "stance_context_fresh_v18_supplement_5000.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v10_part_b.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v11_part_a.csv",
        ROOT
        / "data"
        / "shadow"
        / "stance_context_fresh_v19_supplement"
        / "stance_context_fresh_v19_supplement_5000.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v11_part_b.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_a.csv",
    ]
    paths.extend(
        ROOT / "data" / "shadow" / f"stance_locked_audit_v{version}.csv"
        for version in range(1, 8)
    )
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        values = pd.read_csv(path, encoding="utf-8-sig", usecols=["text_sha256"])
        excluded.update(values["text_sha256"].dropna().astype(str))
    return excluded


def _selected_row(row: dict[str, str]) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    output["rule_stance_label"] = row.get("stance_label", "")
    output["rule_stance_polarity"] = row.get("stance_polarity", "0")
    output["rule_stance_confidence"] = row.get("stance_confidence", "")
    output["rule_stance_cues"] = row.get("stance_cues", "")
    return output


def _selection_bucket(row: dict[str, str]) -> str:
    return "cue_rich" if _DIRECTIONAL_CUE.search(str(row.get("text_excerpt", ""))) else "general"


def select_rows(input_path: Path, excluded: set[str], seed: str = DEFAULT_SEED) -> pd.DataFrame:
    heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, object]]]] = {
        (election, bucket): [] for election in ELECTIONS for bucket in QUOTAS
    }
    seen: set[str] = set(excluded)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 500_000 == 0:
                print(f"[fresh selection] {row_number:,} rows", flush=True)
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
            bucket = _selection_bucket(row)
            key = (election, bucket)
            rank = _rank(text_hash, seed)
            item = (-rank, text_hash, _selected_row(row))
            heap = heaps[key]
            quota = QUOTAS[bucket]
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
    records: list[dict[str, object]] = []
    for (election, bucket), heap in heaps.items():
        if len(heap) != QUOTAS[bucket]:
            raise RuntimeError(
                f"insufficient fresh rows for {election} bucket {bucket}: "
                f"{len(heap)} < {QUOTAS[bucket]}"
            )
        records.extend(dict(item[2], selection_bucket=bucket) for item in heap)
    output = pd.DataFrame(records).sort_values(["election_id", "text_sha256"]).reset_index(drop=True)
    if len(output) != 5_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("fresh V16 selection is not 5,000 unique sentences")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "shadow" / "stance_context_fresh_v16",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--output-name", default="stance_context_fresh_v16_5000.csv")
    args = parser.parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_rows.csv"
    final_path = output_dir / args.output_name
    if final_path.exists():
        raise FileExistsError(f"fresh V16 corpus already exists: {final_path}")
    if selected_path.exists():
        selected = pd.read_csv(selected_path, encoding="utf-8-sig").fillna("")
        print("[fresh selection] resuming from selected_rows.csv", flush=True)
    else:
        selected = select_rows(source, _excluded_hashes(), args.seed)
        _atomic_csv(selected, selected_path)
    groups = collect_context(selected, input_path=source)
    output = attach_context(selected, groups)
    if len(output) != 5_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("fresh context output is not 5,000 unique sentences")
    _atomic_csv(output, final_path)
    state = {
        "status": "fresh_confirmatory_corpus_complete",
        "rows": int(len(output)),
        "seed": args.seed,
        "allowed_elections": list(ELECTIONS),
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "source": str(source),
        "source_size": source.stat().st_size,
        "output": str(final_path),
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "target_counts": output["target_type"].value_counts().to_dict(),
        "selection_bucket_counts": output["selection_bucket"].value_counts().to_dict(),
        "polarity_counts": output["rule_stance_polarity"].value_counts().sort_index().to_dict(),
        "with_context_before": int(output["context_before"].fillna("").ne("").sum()),
        "with_context_after": int(output["context_after"].fillna("").ne("").sum()),
    }
    (output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
