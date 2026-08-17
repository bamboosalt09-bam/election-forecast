"""Build public-treatment proxy features from assembly-derived issue outputs.

This is not a news sentiment model. It derives a conservative proxy for how a
candidate is treated in elite public discourse using already-extracted assembly
issue/candidate links, speaker/member-history party context, and manual issue
direction seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presidential_issue_engine.point_in_time import cutoff_dates_as_strings, filter_available_by_election  # noqa: E402
from presidential_issue_engine.election_scope import ELECTION_DATES  # noqa: E402


RESULTS = ROOT / "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
CANDIDATE_ISSUES = ROOT / "data/candidate_issue_link.csv"
CANDIDATE_PROFILE = ROOT / "data/raw/candidate_issue_profile.csv"
MEGA_ATTRIBUTION = ROOT / "data/raw/mega_issue_attribution.csv"
THIRD_PROFILE = ROOT / "data/raw/third_candidate_profile.csv"
PARTY_CONTEXT = ROOT / "data/raw/candidate_party_speech_context.csv"
PARTY_TONE_GAP = ROOT / "data/raw/candidate_party_tone_gap.csv"
OUTPUT = ROOT / "data/raw/candidate_public_treatment.csv"

RISK_ISSUES = {"corruption_integrity", "family_legal_risk", "gaffe_event"}
LEGITIMACY_ISSUES = {"candidate_competence", "economy_growth", "regime_change", "welfare_pension"}
ALTERNATIVE_ISSUES = {"regime_change", "unification_event", "endorsement_event"}


def _safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_value = float(values.max())
    if max_value <= 0:
        return values * 0.0
    return (values / max_value).clip(0.0, 1.0)


def _candidate_base() -> pd.DataFrame:
    results = _safe_read(RESULTS)
    out = (
        results.loc[
            (results["slot"].astype(str) != "alpha")
            & results["is_active_slot"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        ][["election_id", "slot", "candidate_name", "party_name"]]
        .drop_duplicates(["election_id", "slot"])
        .copy()
    )
    return out


def _issue_components() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    issues = filter_available_by_election(
        _safe_read(CANDIDATE_ISSUES),
        ELECTION_DATES,
        source_name="candidate_issue_link",
    )
    if not issues.empty:
        issues = issues[["election_id", "slot", "issue_name", "emphasis_within", "mentions"]].copy()
        issues["emphasis_within"] = pd.to_numeric(issues["emphasis_within"], errors="coerce").fillna(0.0)
        issues["mentions"] = pd.to_numeric(issues["mentions"], errors="coerce").fillna(0.0)
        pieces.append(issues)
    profile = _safe_read(CANDIDATE_PROFILE)
    required = {"election_id", "slot", "issue_name", "association_strength", "confidence"}
    if not profile.empty and required.issubset(profile.columns):
        profile = profile[["election_id", "slot", "issue_name", "association_strength", "confidence"]].copy()
        profile["emphasis_within"] = (
            pd.to_numeric(profile["association_strength"], errors="coerce").fillna(0.0).abs()
            * pd.to_numeric(profile["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
        profile["mentions"] = profile["emphasis_within"]
        pieces.append(profile[["election_id", "slot", "issue_name", "emphasis_within", "mentions"]])
    if not pieces:
        return pd.DataFrame(columns=["election_id", "slot"])
    issues = pd.concat(pieces, ignore_index=True)
    grouped = issues.groupby(["election_id", "slot"], as_index=False).agg(
        total_attention=("emphasis_within", "sum"),
        total_mentions=("mentions", "sum"),
    )
    risk = (
        issues.loc[issues["issue_name"].isin(RISK_ISSUES)]
        .groupby(["election_id", "slot"], as_index=False)["emphasis_within"]
        .sum()
        .rename(columns={"emphasis_within": "risk_attention"})
    )
    legitimacy = (
        issues.loc[issues["issue_name"].isin(LEGITIMACY_ISSUES)]
        .groupby(["election_id", "slot"], as_index=False)["emphasis_within"]
        .sum()
        .rename(columns={"emphasis_within": "legitimacy_attention"})
    )
    alternative = (
        issues.loc[issues["issue_name"].isin(ALTERNATIVE_ISSUES)]
        .groupby(["election_id", "slot"], as_index=False)["emphasis_within"]
        .sum()
        .rename(columns={"emphasis_within": "alternative_attention"})
    )
    out = grouped.merge(risk, on=["election_id", "slot"], how="left")
    out = out.merge(legitimacy, on=["election_id", "slot"], how="left")
    out = out.merge(alternative, on=["election_id", "slot"], how="left")
    for column in ["risk_attention", "legitimacy_attention", "alternative_attention"]:
        out[column] = out[column].fillna(0.0)
    for column in [
        "total_attention",
        "risk_attention",
        "legitimacy_attention",
        "alternative_attention",
        "total_mentions",
    ]:
        out[f"{column}_norm"] = out.groupby("election_id")[column].transform(_normalize)
    return out


def _manual_direction_components() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    profile = _safe_read(CANDIDATE_PROFILE)
    if not profile.empty:
        for row in profile.itertuples(index=False):
            rows.append(
                {
                    "election_id": getattr(row, "election_id", ""),
                    "slot": getattr(row, "slot", ""),
                    "manual_positive": max(0.0, float(getattr(row, "direction", 0.0)))
                    * float(getattr(row, "association_strength", 0.0))
                    * float(getattr(row, "confidence", 0.0)),
                    "manual_negative": max(0.0, -float(getattr(row, "direction", 0.0)))
                    * float(getattr(row, "association_strength", 0.0))
                    * float(getattr(row, "confidence", 0.0)),
                }
            )
    mega = _safe_read(MEGA_ATTRIBUTION)
    if not mega.empty:
        for row in mega.itertuples(index=False):
            if str(getattr(row, "target_type", "")) != "candidate_slot":
                continue
            rows.append(
                {
                    "election_id": getattr(row, "election_id", ""),
                    "slot": getattr(row, "target", ""),
                    "manual_positive": max(0.0, float(getattr(row, "polarity", 0.0)))
                    * float(getattr(row, "weight", 0.0))
                    * float(getattr(row, "confidence", 0.0)),
                    "manual_negative": max(0.0, -float(getattr(row, "polarity", 0.0)))
                    * float(getattr(row, "weight", 0.0))
                    * float(getattr(row, "confidence", 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["election_id", "slot", "manual_positive", "manual_negative"])
    out = pd.DataFrame(rows)
    return out.groupby(["election_id", "slot"], as_index=False).sum()


def _third_components() -> pd.DataFrame:
    third = _safe_read(THIRD_PROFILE)
    if third.empty:
        return pd.DataFrame(columns=["election_id", "slot", "third_viability", "alternative_score_seed"])
    out = third[["election_id", "slot", "viability", "centrist_appeal", "anti_major_party_appeal", "confidence"]].copy()
    for column in ["viability", "centrist_appeal", "anti_major_party_appeal", "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["third_viability"] = out["viability"] * out["confidence"]
    out["alternative_score_seed"] = (
        out["viability"]
        * (0.55 * out["centrist_appeal"] + 0.45 * out["anti_major_party_appeal"])
        * out["confidence"]
    )
    return out[["election_id", "slot", "third_viability", "alternative_score_seed"]]


def build_treatment() -> pd.DataFrame:
    out = _candidate_base()
    out = out.merge(_issue_components(), on=["election_id", "slot"], how="left")
    out = out.merge(_manual_direction_components(), on=["election_id", "slot"], how="left")
    out = out.merge(_third_components(), on=["election_id", "slot"], how="left")
    party = _safe_read(PARTY_CONTEXT)
    if not party.empty:
        out = out.merge(
            party[
                [
                    "election_id",
                    "slot",
                    "candidate_name",
                    "party_context_support",
                    "organization_strength",
                    "outsider_status",
                    "party_elite_fragmentation_score",
                ]
            ],
            on=["election_id", "slot", "candidate_name"],
            how="left",
        )
    tone = _safe_read(PARTY_TONE_GAP)
    if not tone.empty:
        out = out.merge(
            tone[
                [
                    "election_id",
                    "slot",
                    "candidate_name",
                    "same_party_supportive_tone",
                    "cross_party_adverse_tone",
                    "party_tone_contrast",
                    "confidence",
                ]
            ].rename(columns={"confidence": "party_tone_confidence"}),
            on=["election_id", "slot", "candidate_name"],
            how="left",
        )
    for column in [
        "total_attention_norm",
        "risk_attention_norm",
        "legitimacy_attention_norm",
        "alternative_attention_norm",
        "manual_positive",
        "manual_negative",
        "third_viability",
        "alternative_score_seed",
        "party_context_support",
        "organization_strength",
        "outsider_status",
        "party_elite_fragmentation_score",
        "same_party_supportive_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "party_tone_confidence",
    ]:
        out[column] = pd.to_numeric(out.get(column, 0.0), errors="coerce").fillna(0.0)
    out["same_party_supportive_tone_weighted"] = (
        out["same_party_supportive_tone"].clip(0.0, 1.0)
        * out["party_tone_confidence"].clip(0.0, 1.0)
    )
    out["cross_party_adverse_tone_weighted"] = (
        out["cross_party_adverse_tone"].clip(0.0, 1.0)
        * out["party_tone_confidence"].clip(0.0, 1.0)
    )

    out["serious_contender_score"] = (
        0.40 * out["legitimacy_attention_norm"]
        + 0.30 * out["organization_strength"]
        + 0.20 * out["party_context_support"].clip(0.0, 1.0)
        + 0.10 * out["third_viability"]
    ).clip(0.0, 1.0)
    out["legitimacy_score"] = (
        0.50 * out["legitimacy_attention_norm"]
        + 0.30 * out["manual_positive"].clip(0.0, 1.0)
        + 0.20 * out["party_context_support"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["negative_treatment_score"] = (
        0.45 * out["risk_attention_norm"]
        + 0.35 * out["manual_negative"].clip(0.0, 1.0)
        + 0.20 * out["party_elite_fragmentation_score"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["scandal_salience_score"] = out["risk_attention_norm"].clip(0.0, 1.0)
    out["fatigue_score"] = (
        0.50 * out["risk_attention_norm"] + 0.50 * out["total_attention_norm"] * out["manual_negative"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["alternative_score"] = (
        0.45 * out["alternative_attention_norm"]
        + 0.35 * out["alternative_score_seed"]
        + 0.20 * out["outsider_status"]
    ).clip(0.0, 1.0)
    out["protest_vote_score"] = (
        0.55 * out["alternative_score"]
        + 0.45 * out["manual_positive"].clip(0.0, 1.0) * out["outsider_status"]
    ).clip(0.0, 1.0)
    out["ridicule_or_gaffe_score"] = (
        0.60 * out["risk_attention_norm"] * out["outsider_status"]
        + 0.40 * out["party_elite_fragmentation_score"]
    ).clip(0.0, 1.0)
    out["public_treatment_support"] = (
        out["serious_contender_score"]
        + 0.60 * out["legitimacy_score"]
        + 0.35 * out["alternative_score"]
        + 0.20 * out["protest_vote_score"]
        - 0.70 * out["negative_treatment_score"]
        - 0.45 * out["fatigue_score"]
        - 0.35 * out["ridicule_or_gaffe_score"]
    )
    out["public_treatment_support_centered"] = (
        out["public_treatment_support"]
        - out.groupby("election_id")["public_treatment_support"].transform("mean")
    )
    out["available_date"] = out["election_id"].map(cutoff_dates_as_strings(ELECTION_DATES)).fillna("")
    out["confidence"] = (
        0.45
        + 0.20 * out["total_attention_norm"]
        + 0.20 * out["organization_strength"]
        + 0.15 * out["party_context_support"].clip(0.0, 1.0)
    ).clip(0.0, 0.85)
    out["notes"] = "Assembly issue-link plus speaker/member-history treatment proxy; not news or polling"
    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "scandal_salience_score",
        "fatigue_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "same_party_supportive_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "party_tone_confidence",
        "public_treatment_support",
        "public_treatment_support_centered",
        "available_date",
        "confidence",
        "notes",
    ]
    return out[columns].sort_values(["election_id", "slot"])


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = build_treatment()
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved: {OUTPUT}")
    print(out[["election_id", "slot", "candidate_name", "public_treatment_support_centered", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
