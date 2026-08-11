"""Select a balanced 10,000-row analysis slice from the broad shadow corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
TARGET_TYPES = ("person", "party", "government")
SEED = "stance-context-broad-analysis-10000-v1"


def _rank(text_hash: str, namespace: str) -> str:
    return hashlib.sha256(f"{SEED}|{namespace}|{text_hash}".encode("utf-8")).hexdigest()


def _take(
    selected: list[pd.Series],
    used: set[str],
    rows: pd.DataFrame,
    quota: int,
    namespace: str,
) -> None:
    ranked = rows.assign(
        _rank=rows["text_sha256"].astype(str).map(lambda value: _rank(value, namespace))
    ).sort_values("_rank")
    for _, row in ranked.iterrows():
        if quota <= 0:
            break
        text_hash = str(row["text_sha256"])
        if text_hash in used:
            continue
        used.add(text_hash)
        selected.append(row.drop(labels=["_rank"]))
        quota -= 1


def build_slice(frame: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    used: set[str] = set()
    for election in ELECTIONS:
        election_frame = frame.loc[frame["election_id"].eq(election)].copy()
        start = len(selected)
        representative = election_frame.loc[
            election_frame["sample_component"].eq("representative")
        ]
        _take(selected, used, representative, 1_000, f"{election}:representative")

        supplement = election_frame.loc[
            ~election_frame["sample_component"].eq("representative")
        ]
        for target in TARGET_TYPES:
            _take(
                selected,
                used,
                supplement.loc[supplement["target_type"].eq(target)],
                250,
                f"{election}:target:{target}",
            )

        coverage_groups = [
            group
            for _, group in supplement.groupby(
                ["assembly_daesu", "target_type", "issue_name"], dropna=False, sort=True
            )
        ]
        offsets = [0] * len(coverage_groups)
        ordered_groups = [
            group.assign(
                _rank=group["text_sha256"].astype(str).map(
                    lambda value, index=index: _rank(value, f"{election}:coverage:{index}")
                )
            ).sort_values("_rank")
            for index, group in enumerate(coverage_groups)
        ]
        while len(selected) - start < 2_000:
            progressed = False
            for index, group in enumerate(ordered_groups):
                while offsets[index] < len(group):
                    row = group.iloc[offsets[index]].drop(labels=["_rank"])
                    offsets[index] += 1
                    text_hash = str(row["text_sha256"])
                    if text_hash in used:
                        continue
                    used.add(text_hash)
                    selected.append(row)
                    progressed = True
                    break
                if len(selected) - start >= 2_000:
                    break
            if not progressed:
                break

        if len(selected) - start < 2_000:
            _take(
                selected,
                used,
                election_frame,
                2_000 - (len(selected) - start),
                f"{election}:fill",
            )
        if len(selected) - start != 2_000:
            raise RuntimeError(f"{election}: analysis slice is not 2,000 rows")

    output = pd.DataFrame(selected).reset_index(drop=True)
    if len(output) != 10_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("analysis slice is not 10,000 unique rows")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input.resolve(), encoding="utf-8-sig").fillna("")
    output = build_slice(source)
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    state = {
        "rows": int(len(output)),
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "assembly_counts": output["assembly_daesu"].astype(str).value_counts().sort_index().to_dict(),
        "target_counts": output["target_type"].value_counts().to_dict(),
        "issue_counts": output["issue_name"].value_counts().to_dict(),
        "sample_component_counts": output["sample_component"].value_counts().to_dict(),
        "post_2022_rows_present": bool((~output["election_id"].isin(ELECTIONS)).any()),
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
        "source": str(args.input.resolve()),
        "output": str(destination),
    }
    destination.with_suffix(".state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
