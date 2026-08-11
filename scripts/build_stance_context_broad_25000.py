"""Build a broad, deterministic 25,000-sentence shadow stance corpus.

The sample has two purposes that must remain distinguishable:

* a representative min-hash sample for estimating natural label prevalence;
* a coverage supplement spanning elections, assemblies, target types, and issues.

The script only reads the frozen through-2022 Assembly extraction. It does not
read vote outcomes and it never changes the active forecast model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stance_context_5000 import attach_context, collect_context  # noqa: E402


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
TARGET_TYPES = ("person", "party", "government")
ROWS_PER_ELECTION = 5_000
REPRESENTATIVE_PER_ELECTION = 2_500
COVERAGE_TARGET_PER_ELECTION = 4_000
TARGET_FLOOR_PER_ELECTION = 1_000
RANDOM_POOL_SIZE = 8_000
TARGET_POOL_SIZE = 3_000
COVERAGE_CELL_SIZE = 80
DEFAULT_SEED = "stance-context-broad-25000-v1"


def _rank(text_hash: str, seed: str, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{seed}|{namespace}|{text_hash}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _push_smallest(
    heap: list[tuple[int, str, dict[str, object]]],
    quota: int,
    rank: int,
    text_hash: str,
    row: dict[str, object],
) -> None:
    item = (-rank, text_hash, row)
    if len(heap) < quota:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _ordered_rows(
    heap: Iterable[tuple[int, str, dict[str, object]]]
) -> list[dict[str, object]]:
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _existing_shadow_hashes() -> set[str]:
    """Exclude every prior shadow sentence without relying on a hand-maintained list."""

    excluded: set[str] = set()
    for path in sorted((ROOT / "data" / "shadow").rglob("*.csv")):
        try:
            values = pd.read_csv(path, encoding="utf-8-sig", usecols=["text_sha256"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        excluded.update(values["text_sha256"].dropna().astype(str))
    return excluded


def _selected_row(row: dict[str, str]) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    output["rule_stance_label"] = row.get("stance_label", "")
    output["rule_stance_polarity"] = row.get("stance_polarity", "0")
    output["rule_stance_confidence"] = row.get("stance_confidence", "")
    output["rule_stance_cues"] = row.get("stance_cues", "")
    return output


def _add(
    records: list[dict[str, object]],
    selected_hashes: set[str],
    row: dict[str, object],
    component: str,
) -> bool:
    text_hash = str(row.get("text_sha256", ""))
    if not text_hash or text_hash in selected_hashes:
        return False
    selected_hashes.add(text_hash)
    records.append(dict(row, sample_component=component))
    return True


def select_rows(
    input_path: Path,
    excluded: set[str],
    seed: str = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    random_heaps: dict[str, list[tuple[int, str, dict[str, object]]]] = {
        election: [] for election in ELECTIONS
    }
    target_heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, object]]]] = {
        (election, target): [] for election in ELECTIONS for target in TARGET_TYPES
    }
    coverage_heaps: dict[
        tuple[str, str, str, str], list[tuple[int, str, dict[str, object]]]
    ] = defaultdict(list)
    availability: Counter[tuple[str, str, str, str]] = Counter()
    seen: set[str] = set(excluded)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 500_000 == 0:
                print(f"[broad selection] {row_number:,} rows", flush=True)
            election = str(row.get("election_id", ""))
            target = str(row.get("target_type", ""))
            text_hash = str(row.get("text_sha256", ""))
            if (
                election not in ELECTIONS
                or target not in TARGET_TYPES
                or not text_hash
                or text_hash in seen
            ):
                continue
            seen.add(text_hash)
            selected = _selected_row(row)
            assembly = str(row.get("assembly_daesu", "unknown")) or "unknown"
            issue = str(row.get("issue_name", "unknown")) or "unknown"
            coverage_key = (election, assembly, target, issue)
            availability[coverage_key] += 1

            _push_smallest(
                random_heaps[election],
                RANDOM_POOL_SIZE,
                _rank(text_hash, seed, f"random:{election}"),
                text_hash,
                selected,
            )
            _push_smallest(
                target_heaps[(election, target)],
                TARGET_POOL_SIZE,
                _rank(text_hash, seed, f"target:{election}:{target}"),
                text_hash,
                selected,
            )
            _push_smallest(
                coverage_heaps[coverage_key],
                COVERAGE_CELL_SIZE,
                _rank(text_hash, seed, "coverage:" + ":".join(coverage_key)),
                text_hash,
                selected,
            )

    records: list[dict[str, object]] = []
    selected_hashes: set[str] = set()
    for election in ELECTIONS:
        election_start = len(records)
        random_rows = _ordered_rows(random_heaps[election])
        for row in random_rows:
            if len(records) - election_start >= REPRESENTATIVE_PER_ELECTION:
                break
            _add(records, selected_hashes, row, "representative")

        coverage_lists = {
            key: _ordered_rows(heap)
            for key, heap in coverage_heaps.items()
            if key[0] == election
        }
        coverage_offsets = {key: 0 for key in coverage_lists}
        coverage_keys = sorted(coverage_lists)
        while len(records) - election_start < COVERAGE_TARGET_PER_ELECTION:
            progressed = False
            for key in coverage_keys:
                rows = coverage_lists[key]
                offset = coverage_offsets[key]
                while offset < len(rows):
                    row = rows[offset]
                    offset += 1
                    coverage_offsets[key] = offset
                    if _add(records, selected_hashes, row, "coverage"):
                        progressed = True
                        break
                if len(records) - election_start >= COVERAGE_TARGET_PER_ELECTION:
                    break
            if not progressed:
                break

        counts = Counter(str(row.get("target_type", "")) for row in records[election_start:])
        for target in TARGET_TYPES:
            available_target = sum(
                value
                for key, value in availability.items()
                if key[0] == election and key[2] == target
            )
            target_floor = min(TARGET_FLOOR_PER_ELECTION, available_target)
            for row in _ordered_rows(target_heaps[(election, target)]):
                if counts[target] >= target_floor:
                    break
                if _add(records, selected_hashes, row, "target_floor"):
                    counts[target] += 1

        for row in random_rows:
            if len(records) - election_start >= ROWS_PER_ELECTION:
                break
            _add(records, selected_hashes, row, "representative_fill")

        election_count = len(records) - election_start
        if election_count != ROWS_PER_ELECTION:
            raise RuntimeError(f"{election}: selected {election_count}, expected 5,000")
        counts = Counter(str(row.get("target_type", "")) for row in records[election_start:])
        required = {
            target: min(
                TARGET_FLOOR_PER_ELECTION,
                sum(
                    value
                    for key, value in availability.items()
                    if key[0] == election and key[2] == target
                ),
            )
            for target in TARGET_TYPES
        }
        missing = {
            target: {"selected": counts.get(target, 0), "required": required[target]}
            for target in TARGET_TYPES
            if counts.get(target, 0) < required[target]
        }
        if missing:
            raise RuntimeError(f"{election}: target floor failed: {missing}")

    output = pd.DataFrame(records).sort_values(
        ["election_id", "sample_component", "text_sha256"]
    ).reset_index(drop=True)
    if len(output) != 25_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("broad sample is not 25,000 unique sentences")

    diagnostics: dict[str, object] = {
        "available_unique_target_sentences": int(sum(availability.values())),
        "availability_by_election": {
            election: int(sum(value for key, value in availability.items() if key[0] == election))
            for election in ELECTIONS
        },
        "availability_cells": int(len(availability)),
    }
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "shadow" / "stance_context_broad_25000_v1",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_rows.csv"
    final_path = output_dir / "stance_context_broad_25000.csv"
    diagnostic_path = output_dir / "selection_diagnostics.json"
    if final_path.exists():
        raise FileExistsError(f"broad corpus already exists: {final_path}")

    if selected_path.exists():
        selected = pd.read_csv(selected_path, encoding="utf-8-sig").fillna("")
        diagnostics = (
            json.loads(diagnostic_path.read_text(encoding="utf-8"))
            if diagnostic_path.exists()
            else {}
        )
        print("[broad selection] resuming from selected_rows.csv", flush=True)
    else:
        excluded = _existing_shadow_hashes()
        print(f"[broad selection] excluded prior hashes: {len(excluded):,}", flush=True)
        selected, diagnostics = select_rows(source, excluded, args.seed)
        _atomic_csv(selected, selected_path)
        diagnostic_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    groups = collect_context(selected, input_path=source)
    output = attach_context(selected, groups)
    if len(output) != 25_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("broad context output is not 25,000 unique sentences")
    _atomic_csv(output, final_path)

    representative = output.loc[output["sample_component"].eq("representative")]
    state = {
        "status": "broad_shadow_corpus_complete",
        "rows": int(len(output)),
        "representative_rows": int(len(representative)),
        "seed": args.seed,
        "allowed_elections": list(ELECTIONS),
        "post_2022_rows_present": bool((~output["election_id"].isin(ELECTIONS)).any()),
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
        "source": str(source),
        "source_size": source.stat().st_size,
        "output": str(final_path),
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "assembly_counts": output["assembly_daesu"].astype(str).value_counts().sort_index().to_dict(),
        "target_counts": output["target_type"].value_counts().to_dict(),
        "issue_counts": output["issue_name"].value_counts().to_dict(),
        "sample_component_counts": output["sample_component"].value_counts().to_dict(),
        "representative_target_counts": representative["target_type"].value_counts().to_dict(),
        "rule_polarity_counts": output["rule_stance_polarity"].value_counts().sort_index().to_dict(),
        "with_context_before": int(output["context_before"].fillna("").ne("").sum()),
        "with_context_after": int(output["context_after"].fillna("").ne("").sum()),
        **diagnostics,
    }
    (output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
