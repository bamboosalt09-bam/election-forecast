"""Build and score a blind text-only audit of Assembly stance labels.

Sampling is deterministic and stratified by the existing polarity so neutral,
positive, and negative decisions are all testable. The blind file contains no
source label or metadata; the hidden key is retained separately for scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "full_15_22"
    / "assembly_stance_rows_15_22.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_metadata_blind_audit_300"
)

POLARITY_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
VALID_REVIEW_LABELS = {"negative", "neutral", "positive", "unclear"}


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - margin, center + margin]


def _selection_value(seed: str, text_sha256: str, polarity: int) -> int:
    value = f"{seed}|{polarity}|{text_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def sample_rows(input_path: Path, output_dir: Path, per_polarity: int, seed: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    heaps: dict[int, list[tuple[int, int, dict[str, str]]]] = {-1: [], 0: [], 1: []}
    class_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    rows_read = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_read += 1
            try:
                polarity = int(float(row.get("stance_polarity", "0") or 0))
            except ValueError:
                continue
            if polarity not in heaps:
                continue
            class_counts[polarity] += 1
            label_counts[row.get("stance_label", "")] += 1
            text_hash = row.get("text_sha256", "")
            if not text_hash or text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)
            rank = _selection_value(seed, text_hash, polarity)
            item = (-rank, rows_read, row)
            heap = heaps[polarity]
            if len(heap) < per_polarity:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected: list[tuple[int, dict[str, str]]] = []
    for polarity, heap in heaps.items():
        if len(heap) != per_polarity:
            raise RuntimeError(
                f"polarity {polarity} has only {len(heap)} unique sampled rows; "
                f"requested {per_polarity}"
            )
        selected.extend((polarity, item[2]) for item in heap)
    selected.sort(key=lambda item: _selection_value(seed, item[1]["text_sha256"], item[0]))

    blind_path = output_dir / "blind_review.csv"
    key_path = output_dir / "hidden_key.csv"
    blind_fields = ["audit_id", "text_excerpt", "review_label", "review_notes"]
    key_fields = [
        "audit_id",
        "election_id",
        "assembly_daesu",
        "meeting_date",
        "committee",
        "agenda",
        "speaker",
        "issue_name",
        "target_type",
        "target_name",
        "stance_label",
        "stance_polarity",
        "stance_confidence",
        "stance_cues",
        "text_sha256",
    ]
    with blind_path.open("w", encoding="utf-8-sig", newline="") as blind_handle, key_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as key_handle:
        blind_writer = csv.DictWriter(blind_handle, fieldnames=blind_fields)
        key_writer = csv.DictWriter(key_handle, fieldnames=key_fields)
        blind_writer.writeheader()
        key_writer.writeheader()
        for index, (_, row) in enumerate(selected, 1):
            audit_id = f"S{index:03d}"
            blind_writer.writerow(
                {
                    "audit_id": audit_id,
                    "text_excerpt": row.get("text_excerpt", ""),
                    "review_label": "",
                    "review_notes": "",
                }
            )
            key_writer.writerow({"audit_id": audit_id, **{field: row.get(field, "") for field in key_fields[1:]}})

    state = {
        "status": "awaiting_blind_review",
        "input": str(input_path),
        "rows_read": rows_read,
        "unique_text_hashes": len(seen_hashes),
        "sample_size": len(selected),
        "sample_per_polarity": per_polarity,
        "seed": seed,
        "corpus_polarity_counts": {POLARITY_NAMES[key]: class_counts[key] for key in (-1, 0, 1)},
        "corpus_label_counts": dict(sorted(label_counts.items())),
    }
    (output_dir / "sample_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


def score_review(output_dir: Path) -> None:
    from scripts.extract_assembly_stance_rows import classify_stance

    blind_path = output_dir / "blind_review.csv"
    key_path = output_dir / "hidden_key.csv"
    with blind_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reviews = {row["audit_id"]: row for row in csv.DictReader(handle)}
    annotation_path = output_dir / "blind_annotations.csv"
    if annotation_path.exists():
        with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
            annotations = {row["audit_id"]: row for row in csv.DictReader(handle)}
        if set(annotations) != set(reviews):
            raise ValueError("blind annotation IDs do not match the blind sample")
        for audit_id, annotation in annotations.items():
            reviews[audit_id]["review_label"] = annotation.get("review_label", "")
            reviews[audit_id]["review_notes"] = annotation.get("review_notes", "")
    with key_path.open("r", encoding="utf-8-sig", newline="") as handle:
        keys = {row["audit_id"]: row for row in csv.DictReader(handle)}

    missing = [audit_id for audit_id, row in reviews.items() if row["review_label"] not in VALID_REVIEW_LABELS]
    if missing:
        raise ValueError(f"missing or invalid review labels for {len(missing)} rows: {missing[:10]}")

    confusion: Counter[tuple[str, str]] = Counter()
    scored_rows: list[dict[str, str | int]] = []
    for audit_id, review in reviews.items():
        key = keys[audit_id]
        rule_label = POLARITY_NAMES[int(float(key["stance_polarity"]))]
        review_label = review["review_label"]
        comparable = int(review_label != "unclear")
        agreement = int(comparable and rule_label == review_label)
        if comparable:
            confusion[(rule_label, review_label)] += 1
        scored_rows.append(
            {
                **key,
                "text_excerpt": review["text_excerpt"],
                "review_label": review_label,
                "review_notes": review["review_notes"],
                "comparable": comparable,
                "agreement": agreement,
            }
        )

    comparable_rows = [row for row in scored_rows if row["comparable"]]
    labels = ["negative", "neutral", "positive"]
    correct = sum(int(row["agreement"]) for row in comparable_rows)
    reviewed_counts = Counter(str(row["review_label"]) for row in comparable_rows)
    rule_counts = Counter(
        POLARITY_NAMES[int(float(str(row["stance_polarity"])))] for row in comparable_rows
    )
    observed_agreement = correct / len(comparable_rows)
    expected_agreement = sum(
        reviewed_counts[label] * rule_counts[label] for label in labels
    ) / len(comparable_rows) ** 2
    metrics: dict[str, object] = {
        "sample_size": len(scored_rows),
        "comparable_rows": len(comparable_rows),
        "unclear_rows": len(scored_rows) - len(comparable_rows),
        "overall_accuracy": observed_agreement,
        "overall_accuracy_ci95": _wilson_interval(correct, len(comparable_rows)),
        "cohen_kappa": (observed_agreement - expected_agreement) / (1.0 - expected_agreement),
        "confusion_matrix_rule_rows_review_columns": {
            rule: {review: confusion[(rule, review)] for review in labels} for rule in labels
        },
    }
    per_rule: dict[str, object] = {}
    for rule in labels:
        subset = [row for row in comparable_rows if POLARITY_NAMES[int(float(row["stance_polarity"]))] == rule]
        per_rule[rule] = {
            "n": len(subset),
            "agreement": sum(int(row["agreement"]) for row in subset) / len(subset) if subset else None,
            "agreement_ci95": _wilson_interval(
                sum(int(row["agreement"]) for row in subset), len(subset)
            ),
        }
    metrics["per_rule_label"] = per_rule

    class_metrics: dict[str, object] = {}
    for label in labels:
        true_positive = confusion[(label, label)]
        predicted = sum(confusion[(label, reviewed)] for reviewed in labels)
        reviewed = sum(confusion[(rule, label)] for rule in labels)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / reviewed if reviewed else 0.0
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    metrics["class_metrics"] = class_metrics
    metrics["macro_f1"] = sum(float(value["f1"]) for value in class_metrics.values()) / len(labels)

    directional_correct = sum(
        int(
            (POLARITY_NAMES[int(float(str(row["stance_polarity"])))] != "neutral")
            == (row["review_label"] != "neutral")
        )
        for row in comparable_rows
    )
    both_directional = [
        row
        for row in comparable_rows
        if POLARITY_NAMES[int(float(str(row["stance_polarity"])))] != "neutral"
        and row["review_label"] != "neutral"
    ]
    metrics["directional_vs_neutral_accuracy"] = directional_correct / len(comparable_rows)
    metrics["same_sign_given_both_directional"] = sum(
        int(POLARITY_NAMES[int(float(str(row["stance_polarity"])))] == row["review_label"])
        for row in both_directional
    ) / len(both_directional)
    metrics["both_directional_rows"] = len(both_directional)

    metrics["accuracy_by_rule_detail"] = {}
    for detail in sorted({str(row["stance_label"]) for row in comparable_rows}):
        subset = [row for row in comparable_rows if row["stance_label"] == detail]
        metrics["accuracy_by_rule_detail"][detail] = {
            "n": len(subset),
            "accuracy": sum(int(row["agreement"]) for row in subset) / len(subset),
        }

    state_path = output_dir / "sample_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    corpus_counts = state["corpus_polarity_counts"]
    corpus_total = sum(int(corpus_counts[label]) for label in labels)
    metrics["corpus_weighted_accuracy_estimate"] = sum(
        int(corpus_counts[label]) / corpus_total * float(per_rule[label]["agreement"])
        for label in labels
    )

    recomputed = [classify_stance(str(row["text_excerpt"])) for row in scored_rows]
    exact_matches = sum(
        int(
            result[0] == row["stance_label"]
            and result[1] == int(float(str(row["stance_polarity"])))
            and abs(result[2] - float(str(row["stance_confidence"]))) < 1e-12
            and result[3] == row["stance_cues"]
        )
        for row, result in zip(scored_rows, recomputed, strict=True)
    )
    metadata_fields = [
        "election_id",
        "assembly_daesu",
        "meeting_date",
        "committee",
        "agenda",
        "speaker",
        "issue_name",
        "target_type",
        "target_name",
    ]
    permuted_results: list[tuple[str, int, float, str]] = []
    for index, row in enumerate(scored_rows):
        permuted = dict(row)
        donor = scored_rows[(index + 137) % len(scored_rows)]
        for field in metadata_fields:
            permuted[field] = donor[field]
        permuted_results.append(classify_stance(str(permuted["text_excerpt"])))
    metrics["metadata_invariance"] = {
        "classifier_parameters": list(inspect.signature(classify_stance).parameters),
        "recomputed_exact_matches": exact_matches,
        "sample_size": len(scored_rows),
        "metadata_columns_passed_to_classifier": [],
        "permuted_metadata_columns": metadata_fields,
        "metadata_permutation_changes": sum(
            int(original != permuted)
            for original, permuted in zip(recomputed, permuted_results, strict=True)
        ),
    }

    metadata_breakdown: dict[str, object] = {}
    for field in ("target_type", "assembly_daesu", "issue_name"):
        values: dict[str, object] = {}
        for value in sorted({str(row[field]) for row in comparable_rows}):
            subset = [row for row in comparable_rows if row[field] == value]
            if len(subset) >= 5:
                values[value] = {
                    "n": len(subset),
                    "accuracy": sum(int(row["agreement"]) for row in subset) / len(subset),
                }
        metadata_breakdown[field] = values
    metrics["diagnostic_accuracy_by_metadata_not_causal"] = metadata_breakdown

    fieldnames = list(scored_rows[0].keys())
    with (output_dir / "scored_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    sample_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sample_parser.add_argument("--per-polarity", type=int, default=100)
    sample_parser.add_argument("--seed", default="stance-metadata-audit-v1")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sample":
        sample_rows(args.input, args.output_dir, args.per_polarity, args.seed)
    else:
        score_review(args.output_dir)


if __name__ == "__main__":
    main()
