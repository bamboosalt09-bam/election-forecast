"""Build a frozen 500-row target-aware stance annotation batch."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from election_forecast.stance_target_policy import generic_legacy_label  # noqa: E402
from scripts.evaluate_raw_stance_shadow import candidate_reference  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
V2_DIR = DATA_DIR / "target_aware_v2_protocols"
OUTPUT_DIR = DATA_DIR / "target_aware_annotation_500"
RISK_RE = re.compile(
    r"[‘’“”\"']|라고|다고|라는|이라며|말했|주장|보도|전했|회신|"
    r"비판한다고|지지한다고|찬성한다고|반대한다고|예컨대|예를 들어"
)


def _stable_hash(*values: object) -> str:
    payload = "\x1f".join(str(value or "") for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _current_target_maps() -> dict[str, set[tuple[str, str]]]:
    candidates = candidate_reference()
    return {
        "person": {
            (str(row.election_id), str(row.candidate_name))
            for row in candidates.itertuples(index=False)
        },
        "party": {
            (str(row.election_id), str(row.party_name))
            for row in candidates.itertuples(index=False)
        },
    }


def _select_bucket(group: pd.DataFrame, count: int) -> pd.DataFrame:
    group = group.copy()
    group["selection_priority"] = (
        8 * group["is_current_contest_target"]
        + 5 * group["model_legacy_disagreement"]
        + 3 * group["quote_report_risk"]
        + 2 * group["low_margin_case"]
    )
    # Preserve class diversity before filling by priority. Each class gets up
    # to one third of the bucket, with unused capacity returned to the pool.
    selected_indices: list[int] = []
    per_class = count // 3
    for label in ("negative", "neutral", "positive"):
        candidates = group.loc[group["model_label"].eq(label)].sort_values(
            ["selection_priority", "selection_hash"], ascending=[False, True]
        )
        selected_indices.extend(candidates.head(per_class).index.tolist())
    remaining = group.drop(index=set(selected_indices)).sort_values(
        ["selection_priority", "selection_hash"], ascending=[False, True]
    )
    selected_indices.extend(remaining.head(count - len(selected_indices)).index.tolist())
    return group.loc[selected_indices].copy()


def main() -> None:
    source = pd.read_csv(
        V2_DIR / "target_v2_conservative_labels_5000.csv", encoding="utf-8-sig"
    ).fillna("")
    source = source.loc[source["target_type"].isin(["person", "party"])].copy()
    source["legacy_generic_label"] = source["aggregator_stance_label"].map(
        generic_legacy_label
    )
    current_maps = _current_target_maps()
    source["is_current_contest_target"] = [
        int((str(election_id), str(target_name)) in current_maps[str(target_type)])
        for election_id, target_type, target_name in zip(
            source["election_id"], source["target_type"], source["target_name"], strict=True
        )
    ]
    source["model_legacy_disagreement"] = (
        source["model_label"].astype(str) != source["legacy_generic_label"].astype(str)
    ).astype(int)
    source["quote_report_risk"] = source["text_excerpt"].astype(str).str.contains(RISK_RE).astype(int)
    source["low_margin_case"] = (
        pd.to_numeric(source["model_margin"], errors="coerce").fillna(0.0) < 0.20
    ).astype(int)
    source["selection_hash"] = [
        _stable_hash(election_id, target_type, text_hash, target_name)
        for election_id, target_type, text_hash, target_name in zip(
            source["election_id"],
            source["target_type"],
            source["text_sha256"],
            source["target_name"],
            strict=True,
        )
    ]

    pieces: list[pd.DataFrame] = []
    for election_id, election in source.groupby("election_id", sort=True):
        person_available = int(election["target_type"].eq("person").sum())
        person_count = min(50, person_available)
        party_count = 100 - person_count
        if int(election["target_type"].eq("party").sum()) < party_count:
            raise RuntimeError(f"{election_id}: insufficient party target rows")
        pieces.append(_select_bucket(election.loc[election["target_type"].eq("person")], person_count))
        pieces.append(_select_bucket(election.loc[election["target_type"].eq("party")], party_count))
    selected = pd.concat(pieces, ignore_index=True)
    if len(selected) != 500 or selected["text_sha256"].nunique() != 500:
        raise RuntimeError(
            f"expected 500 unique hashes, found rows={len(selected)}, "
            f"hashes={selected['text_sha256'].nunique()}"
        )

    selected["split_hash"] = [
        _stable_hash("target-aware-v2-split", text_hash)
        for text_hash in selected["text_sha256"]
    ]
    selected["evaluation_split"] = "development"
    for _, indices in selected.groupby(["election_id", "target_type"], sort=True).groups.items():
        ordered = selected.loc[list(indices)].sort_values("split_hash")
        holdout_count = round(len(ordered) * 0.30)
        selected.loc[ordered.head(holdout_count).index, "evaluation_split"] = "frozen_holdout"

    selected.insert(0, "annotation_id", [f"TA{i:04d}" for i in range(1, 501)])
    selected["review_target_present"] = ""
    selected["review_stance_holder"] = ""
    selected["review_polarity"] = ""
    selected["review_intensity_0_3"] = ""
    selected["review_neutral_information_0_3"] = ""
    selected["review_confidence"] = ""
    selected["review_notes"] = ""

    columns = [
        "annotation_id",
        "evaluation_split",
        "election_id",
        "meeting_date",
        "assembly_daesu",
        "committee",
        "agenda",
        "speaker",
        "target_type",
        "target_name",
        "target_alias",
        "issue_name",
        "context_before",
        "text_excerpt",
        "context_after",
        "legacy_generic_label",
        "model_label",
        "model_probability",
        "model_margin",
        "target_v2_label",
        "target_v2_reason",
        "is_current_contest_target",
        "model_legacy_disagreement",
        "quote_report_risk",
        "low_margin_case",
        "text_sha256",
        "source_file",
        "source_row_id",
        "sentence_index",
        "review_target_present",
        "review_stance_holder",
        "review_polarity",
        "review_intensity_0_3",
        "review_neutral_information_0_3",
        "review_confidence",
        "review_notes",
    ]
    selected = selected[columns].sort_values("annotation_id")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_DIR / "target_aware_annotation_500.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "ready_for_blind_review",
        "rows": int(len(selected)),
        "unique_hashes": int(selected["text_sha256"].nunique()),
        "development_rows": int(selected["evaluation_split"].eq("development").sum()),
        "frozen_holdout_rows": int(selected["evaluation_split"].eq("frozen_holdout").sum()),
        "current_contest_target_rows": int(selected["is_current_contest_target"].sum()),
        "model_legacy_disagreement_rows": int(selected["model_legacy_disagreement"].sum()),
        "quote_report_risk_rows": int(selected["quote_report_risk"].sum()),
        "by_election": selected["election_id"].value_counts().sort_index().to_dict(),
        "by_target_type": selected["target_type"].value_counts().to_dict(),
        "review_fields_blank": True,
        "outcomes_used": False,
    }
    (OUTPUT_DIR / "state.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
