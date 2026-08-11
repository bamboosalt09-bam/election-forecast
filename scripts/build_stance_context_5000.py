"""Build a deterministic 5,000-row context dataset for stance modeling."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_INPUT = (
    ROOT / "outputs" / "assembly_stance" / "full_15_22" / "assembly_stance_rows_15_22.csv"
)
AUDIT_KEY = ROOT / "outputs" / "assembly_stance" / "stance_metadata_blind_audit_300" / "hidden_key.csv"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
QUOTAS = {-1: 220, 0: 700, 1: 80}
SEED = "stance-context-5000-v1"


def _rank(text_hash: str) -> str:
    return hashlib.sha256(f"{SEED}|{text_hash}".encode("utf-8")).hexdigest()


def _group_key(row: dict[str, object] | pd.Series) -> tuple[str, str, str]:
    return (
        str(row.get("assembly_daesu", "")),
        str(row.get("source_file", "")),
        str(row.get("source_row_id", "")),
    )


def select_rows() -> pd.DataFrame:
    audit_hashes = set(
        pd.read_csv(AUDIT_KEY, encoding="utf-8-sig", usecols=["text_sha256"])["text_sha256"]
        .dropna()
        .astype(str)
    )
    pieces: list[pd.DataFrame] = []
    for election_id in ELECTIONS:
        source = (
            ROOT
            / "outputs"
            / "assembly_stance"
            / f"pilot_{election_id}_5000"
            / "review_batch.csv"
        )
        frame = pd.read_csv(source, encoding="utf-8-sig")
        frame = frame.loc[~frame["text_sha256"].astype(str).isin(audit_hashes)].copy()
        frame = frame.drop_duplicates("text_sha256")
        for polarity, quota in QUOTAS.items():
            subset = frame.loc[
                pd.to_numeric(frame["rule_stance_polarity"], errors="coerce").fillna(0).eq(polarity)
            ].copy()
            if len(subset) < quota:
                raise RuntimeError(
                    f"{election_id} polarity {polarity}: need {quota}, found {len(subset)}"
                )
            subset["selection_rank"] = subset["text_sha256"].astype(str).map(_rank)
            pieces.append(subset.sort_values("selection_rank").head(quota))
    selected = pd.concat(pieces, ignore_index=True)
    if len(selected) != 5_000 or selected["text_sha256"].duplicated().any():
        raise RuntimeError("context sample is not 5,000 unique sentences")
    return selected


def collect_context(
    selected: pd.DataFrame,
    input_path: Path = FULL_INPUT,
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    wanted = {_group_key(row) for _, row in selected.iterrows()}
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[tuple[str, str, str], str]] = set()
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if row_number % 500_000 == 0:
                print(f"[context scan] {row_number:,} rows", flush=True)
            key = _group_key(row)
            if key not in wanted:
                continue
            text_hash = str(row.get("text_sha256", ""))
            dedup_key = (key, text_hash)
            if not text_hash or dedup_key in seen:
                continue
            seen.add(dedup_key)
            groups[key].append(
                {
                    "sentence_index": int(float(row.get("sentence_index", "0") or 0)),
                    "text_sha256": text_hash,
                    "text_excerpt": str(row.get("text_excerpt", "")),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda row: (int(row["sentence_index"]), str(row["text_sha256"])))
    return groups


def attach_context(
    selected: pd.DataFrame,
    groups: dict[tuple[str, str, str], list[dict[str, object]]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        key = _group_key(selected_row)
        candidates = groups.get(key, [])
        current_index = int(float(selected_row.get("sentence_index", 0) or 0))
        current_hash = str(selected_row["text_sha256"])
        before = [row for row in candidates if int(row["sentence_index"]) < current_index]
        after = [row for row in candidates if int(row["sentence_index"]) > current_index]
        previous = before[-1] if before else None
        following = after[0] if after else None
        found_current = any(str(row["text_sha256"]) == current_hash for row in candidates)
        record = selected_row.to_dict()
        record.update(
            {
                "context_before": previous["text_excerpt"] if previous else "",
                "context_after": following["text_excerpt"] if following else "",
                "context_gap_before": current_index - int(previous["sentence_index"]) if previous else "",
                "context_gap_after": int(following["sentence_index"]) - current_index if following else "",
                "context_current_found": int(found_current),
                "context_group_sentence_count": len(candidates),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_rows()
    groups = collect_context(selected)
    output = attach_context(selected, groups)
    output_path = OUTPUT_DIR / "stance_context_5000.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    state = {
        "status": "complete",
        "rows": int(len(output)),
        "unique_text_hashes": int(output["text_sha256"].nunique()),
        "seed": SEED,
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "polarity_counts": {
            str(key): int(value)
            for key, value in output["rule_stance_polarity"].value_counts().sort_index().items()
        },
        "current_sentence_found": int(output["context_current_found"].sum()),
        "with_context_before": int(output["context_before"].fillna("").ne("").sum()),
        "with_context_after": int(output["context_after"].fillna("").ne("").sum()),
        "source": str(FULL_INPUT),
        "output": str(output_path),
    }
    (OUTPUT_DIR / "dataset_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
