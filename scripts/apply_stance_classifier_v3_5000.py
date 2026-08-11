"""Apply stance v3 and compile a bounded issue-weight overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from election_forecast.stance_v3 import (  # noqa: E402
    classify_issue_character,
    compose_v3_input,
    directional_abstention,
    ownership_abstention,
    temperature_scale,
)
from scripts.evaluate_raw_stance_shadow import candidate_reference  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
V3_DIR = DATA_DIR / "stance_v3"
ACTIVE_OVERLAY = ROOT / "data" / "raw" / "assembly_issue_character_overlay.csv"
LABELS = np.asarray(["negative", "neutral", "positive"])


def _bounded_log(value: float, cap: float) -> float:
    if value <= 0.0:
        return 0.0
    return float(min(np.log1p(value) / np.log1p(cap), 1.0))


def _candidate_maps() -> dict[str, dict[tuple[str, str], str]]:
    candidates = candidate_reference()
    return {
        "person": {
            (str(row.election_id), str(row.candidate_name)): str(row.slot)
            for row in candidates.itertuples(index=False)
        },
        "party": {
            (str(row.election_id), str(row.party_name)): str(row.slot)
            for row in candidates.itertuples(index=False)
        },
    }


def _compile_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    unique = frame.drop_duplicates(["election_id", "issue_name", "text_sha256"]).copy()
    global_rows: list[dict[str, object]] = []
    for (election_id, issue_name), group in unique.groupby(["election_id", "issue_name"], sort=True):
        confidence = float(group["v3_max_probability"].mean())
        confidence_quality = float(np.clip((confidence - 1.0 / 3.0) / (2.0 / 3.0), 0.0, 1.0))
        probability_negative_share = float(group["v3_probability_negative"].mean())
        probability_neutral_share = float(group["v3_probability_neutral"].mean())
        probability_positive_share = float(group["v3_probability_positive"].mean())
        label_mass = group.groupby("v3_label")["v3_max_probability"].sum()
        label_total = float(label_mass.sum())
        negative_share = float(label_mass.get("negative", 0.0) / label_total)
        neutral_share = float(label_mass.get("neutral", 0.0) / label_total)
        positive_share = float(label_mass.get("positive", 0.0) / label_total)
        character = classify_issue_character(
            negative_share,
            neutral_share,
            positive_share,
            confidence_quality=confidence_quality,
        )
        raw_strength = (
            0.55 * _bounded_log(len(group), 500.0)
            + 0.25 * _bounded_log(group["speaker"].astype(str).nunique(), 100.0)
            + 0.10 * _bounded_log(group["committee"].astype(str).nunique(), 12.0)
            + 0.10 * confidence_quality
        )
        global_rows.append(
            {
                "election_id": election_id,
                "issue_name": issue_name,
                "issue_evidence_count": int(len(group)),
                "issue_speaker_count": int(group["speaker"].astype(str).nunique()),
                "issue_confidence_quality": confidence_quality,
                "issue_raw_strength": raw_strength,
                "negative_share": negative_share,
                "neutral_share": neutral_share,
                "positive_share": positive_share,
                "probability_negative_share": probability_negative_share,
                "probability_neutral_share": probability_neutral_share,
                "probability_positive_share": probability_positive_share,
                **character,
                "issue_available_date": pd.to_datetime(group["meeting_date"], errors="coerce").max(),
            }
        )
    global_frame = pd.DataFrame(global_rows)
    global_frame["character_multiplier_raw"] = global_frame["character_multiplier"]
    character_mean = global_frame.groupby("election_id")["character_multiplier_raw"].transform(
        "mean"
    )
    global_frame["character_multiplier"] = (
        global_frame["character_multiplier_raw"] / character_mean
    ).clip(0.88, 1.24)
    global_frame["issue_percentile"] = global_frame.groupby("election_id")["issue_raw_strength"].rank(
        method="average", pct=True
    )
    centered = 2.0 * (global_frame["issue_percentile"] - 0.5)
    global_frame["attention_multiplier"] = (
        1.0
        + 0.10
        * centered
        * (0.5 + 0.5 * global_frame["issue_confidence_quality"])
    ).clip(0.92, 1.08)
    global_frame["attention_character_multiplier"] = (
        global_frame["attention_multiplier"] * global_frame["character_multiplier"]
    ).clip(0.88, 1.24)
    # Forecasting uses issue character, not raw mention volume. Attention remains
    # available below as a diagnostic and ablation-only signal.
    global_frame["salience_multiplier"] = global_frame["character_multiplier"].clip(
        0.88, 1.24
    )

    maps = _candidate_maps()
    target = frame.loc[frame["target_type"].isin(["person", "party"])].copy()
    target["slot"] = [
        maps[str(target_type)].get((str(election_id), str(target_name)), "")
        for election_id, target_type, target_name in zip(
            target["election_id"], target["target_type"], target["target_name"], strict=True
        )
    ]
    target = target.loc[target["slot"].ne("")].drop_duplicates(
        ["election_id", "issue_name", "slot", "target_type", "target_name", "text_sha256"]
    )
    target["signed_evidence"] = target["v3_label"].map(
        {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    ) * target["v3_max_probability"]
    target["absolute_evidence"] = target["signed_evidence"].abs()
    link_rows: list[dict[str, object]] = []
    for (election_id, issue_name, slot), group in target.groupby(
        ["election_id", "issue_name", "slot"], sort=True
    ):
        mass = float(group["absolute_evidence"].sum())
        consistency = abs(float(group["signed_evidence"].sum())) / mass if mass > 0.0 else 0.0
        coverage = _bounded_log(len(group), 20.0)
        confidence = float(group["v3_max_probability"].mean())
        reliability = (0.65 * consistency + 0.35 * coverage) * confidence
        link_rows.append(
            {
                "election_id": election_id,
                "issue_name": issue_name,
                "slot": slot,
                "link_evidence_count": int(len(group)),
                "link_consistency": consistency,
                "link_reliability": reliability,
                "link_reliability_base": reliability,
                "link_available_date": pd.to_datetime(group["meeting_date"], errors="coerce").max(),
            }
        )
    link_frame = pd.DataFrame(link_rows)

    candidates = candidate_reference()[["election_id", "slot"]].drop_duplicates()
    overlay = global_frame.merge(candidates, on="election_id", how="inner")
    if not link_frame.empty:
        overlay = overlay.merge(
            link_frame,
            on=["election_id", "issue_name", "slot"],
            how="left",
        )
    overlay["link_reliability_base"] = pd.to_numeric(
        overlay.get("link_reliability_base", 0.0), errors="coerce"
    ).fillna(0.0)
    overlay["link_multiplier_raw"] = (
        1.0
        + 0.04
        * overlay["link_reliability_base"]
        * (0.5 + 0.5 * overlay["directional_share"])
    ).clip(1.0, 1.04)
    link_mean = overlay.groupby(["election_id", "issue_name"])[
        "link_multiplier_raw"
    ].transform("mean")
    overlay["link_multiplier"] = (overlay["link_multiplier_raw"] / link_mean).clip(
        0.96, 1.04
    )
    overlay["link_evidence_count"] = pd.to_numeric(
        overlay.get("link_evidence_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    overlay["link_consistency"] = pd.to_numeric(
        overlay.get("link_consistency", 0.0), errors="coerce"
    ).fillna(0.0)
    overlay["link_reliability"] = pd.to_numeric(
        overlay.get("link_reliability", 0.0), errors="coerce"
    ).fillna(0.0)
    link_dates = pd.to_datetime(overlay.get("link_available_date"), errors="coerce")
    issue_dates = pd.to_datetime(overlay["issue_available_date"], errors="coerce")
    overlay["available_date"] = pd.concat([issue_dates, link_dates], axis=1).max(axis=1)
    overlay["available_date"] = overlay["available_date"].dt.strftime("%Y-%m-%d")
    return overlay[
        [
            "election_id",
            "issue_name",
            "slot",
            "salience_multiplier",
            "link_multiplier",
            "available_date",
            "issue_evidence_count",
            "issue_speaker_count",
            "issue_confidence_quality",
            "issue_raw_strength",
            "issue_percentile",
            "attention_multiplier",
            "attention_character_multiplier",
            "character_multiplier",
            "character_multiplier_raw",
            "issue_character",
            "character_score",
            "character_intensity",
            "informational_score",
            "accountability_score",
            "performance_score",
            "polarized_score",
            "mixed_score",
            "negative_share",
            "neutral_share",
            "positive_share",
            "probability_negative_share",
            "probability_neutral_share",
            "probability_positive_share",
            "directional_share",
            "directional_balance",
            "polarization",
            "link_evidence_count",
            "link_consistency",
            "link_reliability",
            "link_multiplier_raw",
        ]
    ].sort_values(["election_id", "issue_name", "slot"])


def main() -> None:
    source = pd.read_csv(DATA_DIR / "stance_context_5000.csv", encoding="utf-8-sig").fillna("")
    if source["text_sha256"].duplicated().any():
        raise RuntimeError("v3 application input contains duplicate text hashes")
    artifact = joblib.load(V3_DIR / "stance_context_v3.joblib")
    mode = str(artifact["representation_mode"])
    inputs = pd.Series([compose_v3_input(row, mode) for row in source.to_dict(orient="records")])
    model = artifact["model"]
    raw = model.predict_proba(inputs.astype(str))
    positions = {label: index for index, label in enumerate(model.named_steps["classifier"].classes_)}
    probabilities = np.column_stack([raw[:, positions[label]] for label in LABELS])
    probabilities = temperature_scale(probabilities, float(artifact["temperature"]))
    labels = directional_abstention(
        probabilities,
        LABELS,
        min_probability=float(artifact["min_probability"]),
        min_margin=float(artifact["min_margin"]),
    )
    reasons: list[str] = []
    for index, text in enumerate(source["text_excerpt"]):
        labels[index], reason = ownership_abstention(text, str(labels[index]))
        reasons.append(reason)
    output = source.copy()
    output["v3_label"] = labels
    output["v3_polarity"] = pd.Series(labels).map(
        {"negative": -1, "neutral": 0, "positive": 1}
    ).to_numpy(int)
    output["v3_ownership_reason"] = reasons
    for index, label in enumerate(LABELS):
        output[f"v3_probability_{label}"] = probabilities[:, index]
    output["v3_max_probability"] = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    output["v3_probability_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    output.to_csv(V3_DIR / "v3_predictions_5000.csv", index=False, encoding="utf-8-sig")

    overlay = _compile_overlay(output)
    overlay.to_csv(V3_DIR / "stance_issue_overlay.csv", index=False, encoding="utf-8-sig")
    ACTIVE_OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_csv(ACTIVE_OVERLAY, index=False, encoding="utf-8-sig")
    state = {
        "status": "complete",
        "rows": int(len(output)),
        "unique_hashes": int(output["text_sha256"].nunique()),
        "label_counts": output["v3_label"].value_counts().to_dict(),
        "ownership_abstentions": int(pd.Series(reasons).ne("retained").sum() - pd.Series(reasons).eq("already_neutral").sum()),
        "overlay_rows": int(len(overlay)),
        "salience_multiplier_min": float(overlay["salience_multiplier"].min()),
        "salience_multiplier_max": float(overlay["salience_multiplier"].max()),
        "link_multiplier_min": float(overlay["link_multiplier"].min()),
        "link_multiplier_max": float(overlay["link_multiplier"].max()),
        "direct_candidate_signal_scale_planned": 0.10,
        "issue_character_gain": 0.24,
        "issue_overlay_default_enabled": True,
        "active_overlay_path": str(ACTIVE_OVERLAY.relative_to(ROOT)),
    }
    (V3_DIR / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
