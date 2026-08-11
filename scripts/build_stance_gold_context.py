"""Attach source-row context to the 273 comparable blind stance annotations."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_INPUT = (
    ROOT / "outputs" / "assembly_stance" / "full_15_22" / "assembly_stance_rows_15_22.csv"
)
AUDIT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_metadata_blind_audit_300"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"


def _group_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("assembly_daesu", "")),
        str(row.get("source_file", "")),
        str(row.get("source_row_id", "")),
    )


def main() -> None:
    blind = pd.read_csv(AUDIT_DIR / "blind_review.csv", encoding="utf-8-sig")
    annotations = pd.read_csv(AUDIT_DIR / "blind_annotations.csv", encoding="utf-8-sig")
    key = pd.read_csv(AUDIT_DIR / "hidden_key.csv", encoding="utf-8-sig")
    gold = blind[["audit_id", "text_excerpt"]].merge(
        annotations[["audit_id", "review_label"]], on="audit_id", validate="one_to_one"
    ).merge(key, on="audit_id", validate="one_to_one", suffixes=("", "_key"))
    gold = gold.loc[gold["review_label"].isin(["negative", "neutral", "positive"])].copy()
    wanted_hashes = set(gold["text_sha256"].astype(str))

    locations: dict[str, dict[str, str]] = {}
    with FULL_INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 1_000_000 == 0:
                print(f"[gold location scan] {row_number:,} rows", flush=True)
            text_hash = str(row.get("text_sha256", ""))
            if text_hash in wanted_hashes and text_hash not in locations:
                locations[text_hash] = row
    if len(locations) != len(wanted_hashes):
        raise RuntimeError(f"located {len(locations)} of {len(wanted_hashes)} gold texts")

    wanted_groups = {_group_key(row) for row in locations.values()}
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[tuple[str, str, str], str]] = set()
    with FULL_INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 1_000_000 == 0:
                print(f"[gold context scan] {row_number:,} rows", flush=True)
            group_key = _group_key(row)
            if group_key not in wanted_groups:
                continue
            text_hash = str(row.get("text_sha256", ""))
            dedup_key = (group_key, text_hash)
            if not text_hash or dedup_key in seen:
                continue
            seen.add(dedup_key)
            groups[group_key].append(
                {
                    "sentence_index": int(float(row.get("sentence_index", "0") or 0)),
                    "text_sha256": text_hash,
                    "text_excerpt": str(row.get("text_excerpt", "")),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda row: (int(row["sentence_index"]), str(row["text_sha256"])))

    records: list[dict[str, object]] = []
    for row in gold.to_dict(orient="records"):
        text_hash = str(row["text_sha256"])
        location = locations[text_hash]
        group_key = _group_key(location)
        current_index = int(float(location.get("sentence_index", "0") or 0))
        candidates = groups[group_key]
        before = [candidate for candidate in candidates if int(candidate["sentence_index"]) < current_index]
        after = [candidate for candidate in candidates if int(candidate["sentence_index"]) > current_index]
        previous = before[-1] if before else None
        following = after[0] if after else None
        records.append(
            {
                **row,
                "source_file": location.get("source_file", ""),
                "source_row_id": location.get("source_row_id", ""),
                "sentence_index": current_index,
                "target_alias": location.get("target_alias", ""),
                "context_before": previous["text_excerpt"] if previous else "",
                "context_after": following["text_excerpt"] if following else "",
                "context_gap_before": current_index - int(previous["sentence_index"]) if previous else "",
                "context_gap_after": int(following["sentence_index"]) - current_index if following else "",
            }
        )
    output = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DIR / "gold_context_273.csv", index=False, encoding="utf-8-sig")
    state = {
        "status": "complete",
        "rows": int(len(output)),
        "with_context_before": int(output["context_before"].fillna("").ne("").sum()),
        "with_context_after": int(output["context_after"].fillna("").ne("").sum()),
    }
    (OUTPUT_DIR / "gold_context_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
