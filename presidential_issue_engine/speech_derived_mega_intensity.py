"""Derive election-wide political shock intensity from dated Assembly matches."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCHEMA_VERSION = "speech_derived_mega_intensity_v1"
BASE_INTENSITY = 0.50
INTENSITY_RANGE = 1.50
DOMINANT_ISSUE_SHARE = 0.15
FULL_PHRASE_DENSITY = 1.00
NATIONAL_SPEAKER_COVERAGE = 0.75

OUTPUT_COLUMNS = [
    "election_id",
    "mega_issue_intensity",
    "available_date",
    "notes",
]

# Semantic levels are universal event-class ranks, not election-specific
# coefficients. Numeric severity, persistence, and confidence columns from the
# curated taxonomy are deliberately ignored.
SHOCK_CLASS_LEVEL = {
    "institutional_crisis": 1.00,
    "state_capture_scandal": 0.80,
    "accountability_scandal": 0.60,
    "coalition_realignment": 0.60,
    "incumbent_assessment": 0.60,
    "political_realignment": 0.40,
    "candidate_scandal": 0.40,
    "distributional_policy": 0.20,
}


def _bounded_ratio(value: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return float(np.clip(value / reference, 0.0, 1.0))


def build_automatic_mega_issue_intensity(
    matches: pd.DataFrame,
    election_dates: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return intensity rows and diagnostics using only pre-election evidence.

    The score activates only when the regime-accountability axis is salient,
    phrase-weight dense, and distributed across speakers. Constants represent
    interpretable saturation points and are shared by every election.
    """

    required = {
        "election_id",
        "period",
        "speaker",
        "issue_name",
        "issue_weight",
        "matched_term_count",
    }
    if matches.empty or not required.issubset(matches.columns):
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame()

    frame = matches[list(required)].copy()
    frame["election_id"] = frame["election_id"].astype(str)
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame["issue_weight"] = pd.to_numeric(
        frame["issue_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    frame["matched_term_count"] = pd.to_numeric(
        frame["matched_term_count"], errors="coerce"
    ).fillna(0.0).clip(lower=1.0)
    frame["election_date"] = frame["election_id"].map(election_dates).pipe(
        pd.to_datetime, errors="coerce"
    )
    frame = frame.loc[
        frame["period"].notna()
        & frame["election_date"].notna()
        & frame["period"].lt(frame["election_date"])
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame()

    frame["evidence_mass"] = frame["issue_weight"] * frame["matched_term_count"]
    output_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=True):
        total_mass = float(group["evidence_mass"].sum())
        total_speakers = max(int(group["speaker"].astype(str).nunique()), 1)
        regime = group.loc[group["issue_name"].eq("regime_change")]
        accountability = group.loc[
            group["issue_name"].eq("corruption_integrity")
        ]

        regime_mass = float(regime["evidence_mass"].sum())
        regime_share = regime_mass / total_mass if total_mass > 0.0 else 0.0
        regime_density = regime_mass / max(len(regime), 1)
        regime_speaker_coverage = (
            float(regime["speaker"].astype(str).nunique()) / total_speakers
        )
        accountability_mass = float(accountability["evidence_mass"].sum())
        accountability_share = (
            accountability_mass / total_mass if total_mass > 0.0 else 0.0
        )
        accountability_density = accountability_mass / max(len(accountability), 1)

        salience_component = _bounded_ratio(
            regime_share, DOMINANT_ISSUE_SHARE
        )
        severity_component = _bounded_ratio(
            regime_density, FULL_PHRASE_DENSITY
        )
        breadth_component = _bounded_ratio(
            regime_speaker_coverage, NATIONAL_SPEAKER_COVERAGE
        )
        accountability_component = float(
            np.sqrt(
                _bounded_ratio(accountability_share, DOMINANT_ISSUE_SHARE)
                * _bounded_ratio(accountability_density, FULL_PHRASE_DENSITY)
            )
        )
        corroboration_component = 0.50 + 0.50 * accountability_component
        joint_evidence = float(
            (
                salience_component
                * severity_component
                * breadth_component
                * corroboration_component
            )
            ** 0.25
        )
        intensity = float(
            np.clip(
                BASE_INTENSITY + INTENSITY_RANGE * joint_evidence**2,
                BASE_INTENSITY,
                BASE_INTENSITY + INTENSITY_RANGE,
            )
        )
        available_date = group["period"].max().strftime("%Y-%m-%d")
        output_rows.append(
            {
                "election_id": election_id,
                "mega_issue_intensity": intensity,
                "available_date": available_date,
                "notes": (
                    "Assembly-derived regime-accountability shock; universal "
                    "salience, phrase-density, speaker-breadth formula"
                ),
            }
        )
        diagnostic_rows.append(
            {
                "election_id": election_id,
                "source_rows": int(len(group)),
                "regime_rows": int(len(regime)),
                "regime_share": regime_share,
                "regime_phrase_density": regime_density,
                "regime_speaker_coverage": regime_speaker_coverage,
                "accountability_share": accountability_share,
                "accountability_phrase_density": accountability_density,
                "salience_component": salience_component,
                "severity_component": severity_component,
                "breadth_component": breadth_component,
                "accountability_component": accountability_component,
                "joint_evidence": joint_evidence,
                "mega_issue_intensity": intensity,
                "available_date": available_date,
                "source_model": SCHEMA_VERSION,
            }
        )

    output = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).sort_values(
        "election_id"
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("election_id")
    return output.reset_index(drop=True), diagnostics.reset_index(drop=True)


def gate_intensity_by_event_class(
    diagnostics: pd.DataFrame,
    taxonomy: pd.DataFrame,
    election_dates: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gate speech evidence with dated, categorical event identity only."""

    required = {"election_id", "shock_type", "available_date"}
    if diagnostics.empty or taxonomy.empty or not required.issubset(taxonomy.columns):
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame()
    facts = taxonomy[list(required)].copy()
    facts["election_id"] = facts["election_id"].astype(str)
    facts["available_date"] = pd.to_datetime(facts["available_date"], errors="coerce")
    facts["election_date"] = facts["election_id"].map(election_dates).pipe(
        pd.to_datetime, errors="coerce"
    )
    facts = facts.loc[
        facts["available_date"].notna()
        & facts["election_date"].notna()
        & facts["available_date"].lt(facts["election_date"])
    ].copy()
    facts["event_class_level"] = facts["shock_type"].map(SHOCK_CLASS_LEVEL).fillna(0.0)
    class_summary = (
        facts.sort_values(["election_id", "event_class_level"], ascending=[True, False])
        .drop_duplicates("election_id")
        [["election_id", "shock_type", "event_class_level", "available_date"]]
        .rename(columns={"available_date": "event_available_date"})
    )
    joined = diagnostics.merge(class_summary, on="election_id", how="left")
    joined["event_class_level"] = pd.to_numeric(
        joined["event_class_level"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    joined["mega_issue_intensity"] = (
        BASE_INTENSITY
        + INTENSITY_RANGE
        * joined["joint_evidence"].clip(0.0, 1.0)
        * joined["event_class_level"]
    ).clip(BASE_INTENSITY, BASE_INTENSITY + INTENSITY_RANGE)
    speech_date = pd.to_datetime(joined["available_date"], errors="coerce")
    event_date = pd.to_datetime(joined["event_available_date"], errors="coerce")
    joined["available_date"] = pd.concat([speech_date, event_date], axis=1).max(axis=1)
    joined["available_date"] = joined["available_date"].dt.strftime("%Y-%m-%d")
    output = joined[["election_id", "mega_issue_intensity", "available_date"]].copy()
    output["notes"] = (
        "Assembly regime-accountability evidence gated by dated universal event class"
    )
    joined["source_model"] = SCHEMA_VERSION + "_event_class_gate"
    return output[OUTPUT_COLUMNS].sort_values("election_id"), joined.sort_values(
        "election_id"
    )
