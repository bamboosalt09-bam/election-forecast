"""Build candidate vote-conversion context from assembly-derived candidate signals.

The output is a candidate-level prior layer.  It does not use election outcomes;
it combines assembly-derived public treatment, party speech context, same-party
versus cross-party tone, and third-candidate profile priors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from presidential_issue_engine.election_scope import ELECTION_DATES
    from presidential_issue_engine.point_in_time import cutoff_dates_as_strings
except ModuleNotFoundError:  # supports direct script execution
    from election_scope import ELECTION_DATES
    from point_in_time import cutoff_dates_as_strings


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CANDIDATE_PARTY_SPEECH_CONTEXT = RAW / "candidate_party_speech_context.csv"
CANDIDATE_PARTY_TONE_GAP = RAW / "candidate_party_tone_gap.csv"
CANDIDATE_PUBLIC_TREATMENT = RAW / "candidate_public_treatment.csv"
THIRD_CANDIDATE_PROFILE = RAW / "third_candidate_profile.csv"
OUT = RAW / "candidate_vote_conversion_context.csv"

KEYS = ["election_id", "slot", "candidate_name"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _bounded(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower, upper)


def _candidate_base() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in [
        CANDIDATE_PUBLIC_TREATMENT,
        CANDIDATE_PARTY_SPEECH_CONTEXT,
        CANDIDATE_PARTY_TONE_GAP,
    ]:
        frame = _read(path)
        if not frame.empty and set(KEYS).issubset(frame.columns):
            pieces.append(frame[KEYS].copy())
    if not pieces:
        third = _read(THIRD_CANDIDATE_PROFILE)
        if not third.empty and set(KEYS).issubset(third.columns):
            return third[KEYS].copy().drop_duplicates(KEYS).reset_index(drop=True)
        return pd.DataFrame(columns=KEYS)
    return pd.concat(pieces, ignore_index=True).drop_duplicates(KEYS).reset_index(drop=True)


def build() -> pd.DataFrame:
    base = _candidate_base()
    if base.empty:
        return base

    public = _read(CANDIDATE_PUBLIC_TREATMENT)
    if not public.empty:
        keep = [
            *KEYS,
            "serious_contender_score",
            "legitimacy_score",
            "negative_treatment_score",
            "alternative_score",
            "protest_vote_score",
            "ridicule_or_gaffe_score",
            "public_treatment_support",
            "available_date",
            "confidence",
        ]
        public = public[[column for column in keep if column in public.columns]].copy()
        public = public.rename(
            columns={
                "available_date": "public_available_date",
                "confidence": "public_confidence",
            }
        )
        base = base.merge(public, on=KEYS, how="left")

    party = _read(CANDIDATE_PARTY_SPEECH_CONTEXT)
    if not party.empty:
        keep = [
            *KEYS,
            "party_elite_support_score",
            "party_elite_fragmentation_score",
            "party_context_support",
            "organization_strength",
            "outsider_status",
            "available_date",
            "confidence",
        ]
        party = party[[column for column in keep if column in party.columns]].copy()
        party = party.rename(
            columns={
                "available_date": "party_available_date",
                "confidence": "party_confidence",
            }
        )
        base = base.merge(party, on=KEYS, how="left")

    tone = _read(CANDIDATE_PARTY_TONE_GAP)
    if not tone.empty:
        keep = [
            *KEYS,
            "same_party_supportive_tone",
            "cross_party_positive_tone",
            "cross_party_adverse_tone",
            "party_tone_contrast",
            "available_date",
            "confidence",
        ]
        tone = tone[[column for column in keep if column in tone.columns]].copy()
        tone = tone.rename(
            columns={
                "available_date": "tone_available_date",
                "confidence": "tone_confidence",
            }
        )
        base = base.merge(tone, on=KEYS, how="left")

    third = _read(THIRD_CANDIDATE_PROFILE)
    if not third.empty:
        keep = [
            "election_id",
            "slot",
            "viability",
            "centrist_appeal",
            "anti_major_party_appeal",
            "regional_base_overlap",
            "available_date",
            "confidence",
        ]
        third = third[[column for column in keep if column in third.columns]].copy()
        third = third.rename(
            columns={
                "viability": "third_viability",
                "confidence": "third_profile_confidence",
                "available_date": "third_available_date",
            }
        )
        base = base.merge(third, on=["election_id", "slot"], how="left")

    for column in [
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "public_treatment_support",
        "public_confidence",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "party_context_support",
        "organization_strength",
        "outsider_status",
        "party_confidence",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "tone_confidence",
        "third_viability",
        "centrist_appeal",
        "anti_major_party_appeal",
        "regional_base_overlap",
        "third_profile_confidence",
    ]:
        if column not in base.columns:
            base[column] = 0.0

    serious = _bounded(base["serious_contender_score"])
    legitimacy = _bounded(base["legitimacy_score"])
    organization = _bounded(base["organization_strength"])
    party_support = _bounded(base["party_elite_support_score"])
    fragmentation = _bounded(base["party_elite_fragmentation_score"])
    same_party = _bounded(base["same_party_supportive_tone"])
    cross_positive = _bounded(base["cross_party_positive_tone"])
    outsider = _bounded(base["outsider_status"])
    third_viability = _bounded(base["third_viability"])
    regional_overlap = _bounded(base["regional_base_overlap"])
    centrist = _bounded(base["centrist_appeal"])
    anti_major = _bounded(base["anti_major_party_appeal"])
    confidence = pd.concat(
        [
            _bounded(base["public_confidence"]),
            _bounded(base["party_confidence"]),
            _bounded(base["tone_confidence"]),
            _bounded(base["third_profile_confidence"]),
        ],
        axis=1,
    ).replace(0.0, np.nan).mean(axis=1).fillna(0.0).clip(0.0, 1.0)

    public_support_norm = (
        pd.to_numeric(base["public_treatment_support"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        + 1.0
    ) / 2.0
    alternative = _bounded(base["alternative_score"])
    protest = _bounded(base["protest_vote_score"])
    negative = _bounded(base["negative_treatment_score"])
    ridicule = _bounded(base["ridicule_or_gaffe_score"])

    base["candidate_weight"] = (
        0.25 * serious
        + 0.20 * legitimacy
        + 0.20 * organization
        + 0.15 * party_support
        + 0.10 * third_viability
        + 0.10 * (0.60 * centrist + 0.40 * anti_major)
    ).clip(0.0, 1.0)
    base["coalition_cohesion"] = (
        0.35 * organization
        + 0.25 * party_support
        + 0.20 * same_party
        + 0.20 * (1.0 - fragmentation)
    ).clip(0.0, 1.0)
    base["coalition_mobilization_score"] = (
        0.35 * base["coalition_cohesion"]
        + 0.30 * party_support
        + 0.20 * same_party
        + 0.15 * pd.to_numeric(base["party_context_support"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    base["third_character_constraint"] = np.where(
        base["slot"].astype(str).eq("C"),
        (
            0.45 * outsider
            + 0.25 * (1.0 - organization)
            + 0.20 * (1.0 - regional_overlap)
            + 0.10 * (1.0 - base["coalition_cohesion"])
        )
        * (1.0 - 0.35 * third_viability * regional_overlap),
        0.0,
    ).clip(0.0, 1.0)
    base["candidate_weight"] = (
        base["candidate_weight"] * (1.0 - 0.10 * base["third_character_constraint"])
    ).clip(0.0, 1.0)
    base["wasted_vote_resistance"] = (
        0.30 * base["candidate_weight"]
        + 0.25 * third_viability
        + 0.18 * serious
        + 0.12 * legitimacy
        + 0.18 * base["coalition_mobilization_score"]
        - 0.10 * outsider
        + 0.05 * regional_overlap
        - 0.18 * base["third_character_constraint"]
    ).clip(0.0, 1.0)
    attention_index = (
        0.35 * public_support_norm
        + 0.25 * alternative
        + 0.20 * protest
        + 0.10 * cross_positive
        + 0.10 * anti_major
    ).clip(0.0, 1.0)
    base["attention_to_support_gap"] = (
        attention_index
        - (0.70 * base["candidate_weight"] + 0.30 * base["wasted_vote_resistance"])
    ).clip(lower=0.0, upper=1.0)
    base["third_candidate_overexposure_risk"] = (
        base["attention_to_support_gap"]
        * (1.0 - 0.55 * base["wasted_vote_resistance"])
        * np.where(base["slot"].astype(str).eq("C"), 1.0, 0.35)
        + 0.10 * negative
        + 0.08 * ridicule
        + 0.18 * base["third_character_constraint"]
    ).clip(0.0, 1.0)
    base["conversion_capacity"] = (
        base["candidate_weight"]
        * (0.55 + 0.45 * base["wasted_vote_resistance"])
        * (0.65 + 0.35 * base["coalition_mobilization_score"])
        * (1.0 - 0.22 * base["third_character_constraint"])
        * confidence.clip(lower=0.35)
    ).clip(0.0, 1.0)

    major_by_election = (
        base.loc[base["slot"].astype(str).isin(["A", "B"])]
        .groupby("election_id")["conversion_capacity"]
        .mean()
        .rename("major_conversion_capacity")
    )
    base = base.merge(major_by_election, on="election_id", how="left")
    base["major_conversion_capacity"] = base["major_conversion_capacity"].fillna(0.0)
    base["major_party_gravity"] = np.where(
        base["slot"].astype(str).eq("C"),
        (
            base["major_conversion_capacity"]
            * (1.0 - 0.65 * base["coalition_cohesion"])
            * (1.0 - 0.35 * base["wasted_vote_resistance"])
            * (1.0 + 0.30 * base["third_character_constraint"])
        ).clip(0.0, 1.0),
        0.0,
    )

    date_columns = [
        column
        for column in [
            "public_available_date",
            "party_available_date",
            "tone_available_date",
            "third_available_date",
        ]
        if column in base.columns
    ]
    if date_columns:
        available = base[date_columns].apply(pd.to_datetime, errors="coerce").max(axis=1)
    else:
        available = pd.Series(pd.NaT, index=base.index)
    cutoff = pd.to_datetime(
        base["election_id"].map(cutoff_dates_as_strings(ELECTION_DATES)),
        errors="coerce",
    )
    base["available_date"] = available.fillna(cutoff).dt.date.astype(str)
    base["confidence"] = confidence
    base["notes"] = "Derived from assembly public-treatment, party-context, tone-gap, and third-candidate profile signals; no vote outcomes used"

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "candidate_weight",
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
        "available_date",
        "confidence",
        "notes",
    ]
    out = base[columns].copy()
    for column in columns[3:12] + ["confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out.sort_values(["election_id", "slot", "candidate_name"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW)
    parser.add_argument("--third-candidate-profile", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--target-election", default=None)
    parser.add_argument("--preserve-history-from", type=Path, default=None)
    args = parser.parse_args()

    global CANDIDATE_PARTY_SPEECH_CONTEXT
    global CANDIDATE_PARTY_TONE_GAP
    global CANDIDATE_PUBLIC_TREATMENT
    global THIRD_CANDIDATE_PROFILE
    CANDIDATE_PARTY_SPEECH_CONTEXT = args.input_dir / "candidate_party_speech_context.csv"
    CANDIDATE_PARTY_TONE_GAP = args.input_dir / "candidate_party_tone_gap.csv"
    CANDIDATE_PUBLIC_TREATMENT = args.input_dir / "candidate_public_treatment.csv"
    THIRD_CANDIDATE_PROFILE = (
        args.third_candidate_profile
        if args.third_candidate_profile is not None
        else args.input_dir / "third_candidate_profile.csv"
    )

    out = build()
    if args.target_election:
        target = out.loc[out["election_id"].astype(str).eq(args.target_election)].copy()
        if args.preserve_history_from is not None:
            history = _read(args.preserve_history_from)
            history = history.loc[
                ~history["election_id"].astype(str).eq(args.target_election)
            ].copy()
            out = pd.concat([history, target], ignore_index=True)
        else:
            out = target
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[write] {args.out} rows={len(out)}")
    if not out.empty:
        summary = out.groupby("election_id")[["candidate_weight", "wasted_vote_resistance", "conversion_capacity"]].mean()
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
