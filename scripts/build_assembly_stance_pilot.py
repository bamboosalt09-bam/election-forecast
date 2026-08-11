"""Build a deterministic, review-ready sentence pilot from stance extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "assembly_stance" / "full_15_22" / "assembly_stance_rows_15_22.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "assembly_stance"
REVIEW_COLUMNS = [
    "review_id",
    "sample_bucket",
    "selection_rank",
    "election_id",
    "meeting_date",
    "assembly_daesu",
    "committee",
    "agenda",
    "speaker",
    "source_file",
    "source_row_id",
    "sentence_index",
    "issue_name",
    "issue_weight",
    "target_type",
    "target_name",
    "target_alias",
    "rule_stance_label",
    "rule_stance_polarity",
    "rule_stance_confidence",
    "rule_stance_cues",
    "text_excerpt",
    "text_sha256",
    "review_target_correct",
    "review_target_scope",
    "review_stance_label",
    "review_intensity",
    "review_notes",
]

# The pilot deliberately oversamples sentences that the current conservative
# rule labels as directional, while retaining neutral controls for calibration.
DEFAULT_QUOTAS = {
    "person_directional": 80,
    "party_directional": 80,
    "government_directional": 80,
    "person_neutral": 50,
    "party_neutral": 50,
    "government_neutral": 50,
    "untargeted_neutral": 110,
}

# This profile is for error analysis, not prevalence estimation. Its directional
# quotas intentionally exceed the 2022 supply so rare rule-positive examples
# are retained rather than lost in a mostly neutral random sample.
VALIDATION_3000_QUOTAS = {
    "person_directional": 250,
    "party_directional": 250,
    "government_directional": 500,
    "person_neutral": 400,
    "party_neutral": 200,
    "government_neutral": 500,
    "untargeted_neutral": 900,
}

VALIDATION_5000_QUOTAS = {
    "person_directional": 500,
    "party_directional": 500,
    "government_directional": 750,
    "person_neutral": 750,
    "party_neutral": 400,
    "government_neutral": 800,
    "untargeted_neutral": 1300,
}


def sample_bucket(row: dict[str, str]) -> str:
    target_type = row.get("target_type", "")
    stance = row.get("stance_label", "")
    if target_type in {"person", "party", "government"}:
        return f"{target_type}_{'neutral' if stance == 'neutral' else 'directional'}"
    return "untargeted_neutral" if stance == "neutral" else "other_directional"


def quotas_for(profile: str, sample_size: int) -> dict[str, int]:
    if profile == "review500":
        quotas = dict(DEFAULT_QUOTAS)
    elif profile == "validation3000":
        quotas = dict(VALIDATION_3000_QUOTAS)
    elif profile == "validation5000":
        quotas = dict(VALIDATION_5000_QUOTAS)
    else:
        raise ValueError(f"unsupported profile: {profile}")
    quota_total = sum(quotas.values())
    if sample_size != quota_total:
        quotas["untargeted_neutral"] = max(0, quotas["untargeted_neutral"] + sample_size - quota_total)
    return quotas


def _rank(seed: int, text_hash: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text_hash}".encode("ascii", "ignore")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _pilot_row(row: dict[str, str], bucket: str, rank: int) -> dict[str, str]:
    out = {
        "sample_bucket": bucket,
        "selection_rank": str(rank),
        "election_id": row.get("election_id", ""),
        "meeting_date": row.get("meeting_date", ""),
        "assembly_daesu": row.get("assembly_daesu", ""),
        "committee": row.get("committee", ""),
        "agenda": row.get("agenda", ""),
        "speaker": row.get("speaker", ""),
        "source_file": row.get("source_file", ""),
        "source_row_id": row.get("source_row_id", ""),
        "sentence_index": row.get("sentence_index", ""),
        "issue_name": row.get("issue_name", ""),
        "issue_weight": row.get("issue_weight", ""),
        "target_type": row.get("target_type", ""),
        "target_name": row.get("target_name", ""),
        "target_alias": row.get("target_alias", ""),
        "rule_stance_label": row.get("stance_label", ""),
        "rule_stance_polarity": row.get("stance_polarity", ""),
        "rule_stance_confidence": row.get("stance_confidence", ""),
        "rule_stance_cues": row.get("stance_cues", ""),
        "text_excerpt": row.get("text_excerpt", ""),
        "text_sha256": row.get("text_sha256", ""),
        "review_target_correct": "",
        "review_target_scope": "",
        "review_stance_label": "",
        "review_intensity": "",
        "review_notes": "",
    }
    return out


def select_pilot_rows(
    rows: Iterable[dict[str, str]],
    election_id: str,
    sample_size: int,
    seed: int,
    profile: str = "review500",
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    quotas = quotas_for(profile, sample_size)

    # A max heap stores the lowest deterministic ranks without retaining all
    # candidate rows. The set prevents one multi-issue sentence being sampled
    # more than once.
    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {bucket: [] for bucket in quotas}
    fallback_heap: list[tuple[int, str, dict[str, str]]] = []
    seen_texts: set[str] = set()
    eligible_by_bucket: Counter[str] = Counter()
    unique_candidate_sentences = 0
    for row in rows:
        if row.get("election_id") != election_id:
            continue
        text_hash = row.get("text_sha256", "")
        if not text_hash or text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        unique_candidate_sentences += 1
        bucket = sample_bucket(row)
        rank = _rank(seed, text_hash)
        candidate = _pilot_row(row, bucket, rank)
        fallback_item = (-rank, text_hash, candidate)
        if len(fallback_heap) < sample_size:
            heapq.heappush(fallback_heap, fallback_item)
        elif rank < -fallback_heap[0][0]:
            heapq.heapreplace(fallback_heap, fallback_item)
        if bucket not in quotas:
            continue
        eligible_by_bucket[bucket] += 1
        item = (-rank, text_hash, candidate)
        heap = heaps[bucket]
        if len(heap) < quotas[bucket]:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)

    selected = [item[2] for heap in heaps.values() for item in heap]
    selected_hashes = {row["text_sha256"] for row in selected}
    selected.sort(key=lambda row: (row["sample_bucket"], int(row["selection_rank"])))

    # Sparse strata are filled from the remaining deterministic candidates so
    # every pilot contains the requested number of distinct sentences.
    if len(selected) < sample_size:
        fallback = sorted(((-rank, row) for rank, _, row in fallback_heap), key=lambda item: item[0])
        selected.extend(
            dict(row, sample_bucket="fallback")
            for _, row in fallback
            if row["text_sha256"] not in selected_hashes
        )
        selected = selected[:sample_size]
        selected.sort(key=lambda row: (row["sample_bucket"], int(row["selection_rank"])))

    for index, row in enumerate(selected, 1):
        row["review_id"] = f"{election_id}_{index:03d}"

    metadata = {
        "election_id": election_id,
        "sample_size_requested": sample_size,
        "sample_size_selected": len(selected),
        "seed": seed,
        "profile": profile,
        "unique_candidate_sentences": unique_candidate_sentences,
        "eligible_by_bucket": dict(sorted(eligible_by_bucket.items())),
        "selected_by_bucket": dict(sorted(Counter(row["sample_bucket"] for row in selected).items())),
        "selection_method": "deterministic hash-ranked stratified sample, unique text_sha256",
    }
    return selected, metadata


def build_pilot(
    input_path: Path,
    output_dir: Path,
    election_id: str,
    sample_size: int,
    seed: int,
    profile: str,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"stance input not found: {input_path}")
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows, metadata = select_pilot_rows(csv.DictReader(handle), election_id, sample_size, seed, profile)
    if len(rows) != sample_size:
        raise RuntimeError(f"only selected {len(rows)} unique sentences, expected {sample_size}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "review_batch.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# Assembly Stance Pilot",
        "",
        f"- Election: `{election_id}`",
        f"- Unique review sentences: `{len(rows)}`",
        f"- Sampling seed: `{seed}`",
        f"- Sampling profile: `{profile}`",
        "- Current labels are deterministic rule labels for audit only, not training targets.",
        "",
        "## Composition",
        "",
        "| Bucket | Eligible unique sentences | Selected |",
        "| --- | ---: | ---: |",
    ]
    for bucket in sorted(set(metadata["eligible_by_bucket"]) | set(metadata["selected_by_bucket"])):
        report.append(
            f"| {bucket} | {metadata['eligible_by_bucket'].get(bucket, 0)} | {metadata['selected_by_bucket'].get(bucket, 0)} |"
        )
    report.extend(
        [
            "",
            "## Review Fields",
            "",
            "Annotate `review_target_correct`, `review_target_scope`, `review_stance_label`, "
            "`review_intensity`, and `review_notes` in `review_batch.csv`. Do not use "
            "these rows as model input until the review labels and point-in-time policy "
            "are audited.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "pilot_pres_2022_500")
    parser.add_argument("--election-id", default="pres_2022")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202207)
    parser.add_argument(
        "--profile",
        choices=["review500", "validation3000", "validation5000"],
        default="review500",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_pilot(args.input, args.output_dir, args.election_id, args.sample_size, args.seed, args.profile)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
