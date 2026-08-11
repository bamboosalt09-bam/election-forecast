"""Forecast-safe response to incumbent burden and unusually strong shocks.

The adjustment is deliberately post-model and outcome blind.  It uses only
explicit government-target attribution from the issue profile, prior direct
party ballots, the pre-election point forecast, and PIT-filtered shock size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.point_in_time import filter_available_by_election


DEFAULT_GOVERNMENT_BURDEN_GAIN = 1.0
DEFAULT_RUPTURE_EXTRA_GAIN = 0.40
DEFAULT_CONVERSION_BUFFER = 0.15
DEFAULT_LOG_SHIFT_CAP = 0.15


def compile_government_burden_scores(
    profile: pd.DataFrame,
    election_dates: dict[str, str],
) -> pd.DataFrame:
    """Aggregate explicitly government-targeted directional evidence by candidate."""

    columns = [
        "election_id",
        "slot",
        "government_direction_score",
        "government_evidence_weight",
        "government_evidence_count",
        "government_negative_evidence_mass",
        "government_positive_evidence_mass",
        "government_negative_share",
        "government_negative_issue_count",
        "government_rejection_breadth",
        "government_rejection_strength",
    ]
    required = {
        "election_id",
        "slot",
        "issue_name",
        "direction",
        "association_strength",
        "confidence",
        "target_absolute_evidence",
        "target_attribution_confidence",
        "target_source_types",
        "available_date",
    }
    if profile.empty or not required.issubset(profile.columns):
        return pd.DataFrame(columns=columns)

    frame = filter_available_by_election(
        profile.copy(),
        election_dates,
        source_name="candidate_issue_profile_government_burden",
    )
    for column in [
        "direction",
        "association_strength",
        "confidence",
        "target_absolute_evidence",
        "target_attribution_confidence",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.loc[
        frame["target_source_types"].fillna("").astype(str).str.contains(
            r"(?:^|\|)government(?:\||$)", regex=True
        )
        & frame["direction"].abs().ge(0.10)
        & frame["target_absolute_evidence"].gt(0.0)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["government_evidence_weight"] = (
        frame["target_attribution_confidence"].clip(0.0, 1.0)
        * np.log1p(frame["target_absolute_evidence"].clip(lower=0.0))
    )
    frame["weighted_direction"] = (
        frame["direction"].clip(-1.0, 1.0)
        * frame["association_strength"].clip(0.0, 1.0)
        * frame["confidence"].clip(0.0, 1.0)
        * frame["government_evidence_weight"]
    )
    frame["raw_direction"] = (
        frame["direction"].clip(-1.0, 1.0)
        * frame["association_strength"].clip(0.0, 1.0)
        * frame["confidence"].clip(0.0, 1.0)
    )
    frame["government_negative_evidence_mass"] = np.where(
        frame["raw_direction"].lt(0.0),
        -frame["raw_direction"] * frame["government_evidence_weight"],
        0.0,
    )
    frame["government_positive_evidence_mass"] = np.where(
        frame["raw_direction"].gt(0.0),
        frame["raw_direction"] * frame["government_evidence_weight"],
        0.0,
    )
    frame["negative_issue_name"] = np.where(
        frame["raw_direction"].lt(0.0), frame["issue_name"].astype(str), ""
    )
    grouped = frame.groupby(["election_id", "slot"], as_index=False).agg(
        weighted_direction=("weighted_direction", "sum"),
        government_evidence_weight=("government_evidence_weight", "sum"),
        government_evidence_count=("weighted_direction", "size"),
        government_negative_evidence_mass=("government_negative_evidence_mass", "sum"),
        government_positive_evidence_mass=("government_positive_evidence_mass", "sum"),
        government_negative_issue_count=(
            "negative_issue_name",
            lambda values: int(values.loc[values.ne("")].nunique()),
        ),
    )
    grouped["government_direction_score"] = (
        grouped["weighted_direction"]
        / grouped["government_evidence_weight"].replace(0.0, np.nan)
    ).fillna(0.0).clip(-1.0, 1.0)
    directional_mass = (
        grouped["government_negative_evidence_mass"]
        + grouped["government_positive_evidence_mass"]
    )
    grouped["government_negative_share"] = (
        grouped["government_negative_evidence_mass"]
        / directional_mass.replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    grouped["government_rejection_breadth"] = (
        grouped["government_negative_issue_count"] / 4.0
    ).clip(0.0, 1.0)
    grouped["government_rejection_strength"] = (
        (-grouped["government_direction_score"]).clip(lower=0.0)
        * grouped["government_negative_share"]
        * np.sqrt(grouped["government_rejection_breadth"])
    ).clip(0.0, 1.0)
    return grouped[columns]


def apply_incumbent_shock_response(
    frame: pd.DataFrame,
    burden_scores: pd.DataFrame,
    intensity: pd.DataFrame,
    election_dates: dict[str, str],
    *,
    prediction_column: str,
    slot_column: str = "source_slot",
    output_column: str | None = None,
    government_burden_gain: float = DEFAULT_GOVERNMENT_BURDEN_GAIN,
    rupture_extra_gain: float = DEFAULT_RUPTURE_EXTRA_GAIN,
    conversion_buffer: float = DEFAULT_CONVERSION_BUFFER,
    log_shift_cap: float = DEFAULT_LOG_SHIFT_CAP,
) -> pd.DataFrame:
    """Apply bounded incumbent burden and high-shock response, then renormalize.

    Direct-party strength protects a candidate's durable base.  A candidate
    whose forecast substantially exceeds that party base is also protected as
    a plausible personal/coalitional vote converter.  Consequently the burden
    acts mainly on weak, weakly converting governing-camp candidacies.
    """

    output_column = output_column or prediction_column
    required = {
        "election_id",
        "region_id",
        slot_column,
        prediction_column,
        "direct_party_recent_base",
        "direct_party_reliability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"incumbent shock frame missing columns: {sorted(missing)}")

    out = frame.copy().reset_index(drop=True)
    generated_columns = [
        "government_direction_score",
        "government_evidence_weight",
        "government_evidence_count",
        "government_negative_evidence_mass",
        "government_positive_evidence_mass",
        "government_negative_share",
        "government_negative_issue_count",
        "government_rejection_breadth",
        "government_rejection_strength",
        "mega_issue_intensity_response",
        "party_base_resistance",
        "conversion_resistance",
        "incumbent_burden_resistance",
        "incumbent_burden_exposure",
        "government_burden_log_shift",
        "rupture_extra_log_shift",
        "incumbent_shock_log_shift",
    ]
    out = out.drop(
        columns=[column for column in generated_columns if column in out.columns]
    )
    scores = burden_scores.rename(columns={"slot": slot_column}).copy()
    score_columns = [
        "election_id",
        slot_column,
        "government_direction_score",
        "government_evidence_weight",
        "government_evidence_count",
        "government_negative_evidence_mass",
        "government_positive_evidence_mass",
        "government_negative_share",
        "government_negative_issue_count",
        "government_rejection_breadth",
        "government_rejection_strength",
    ]
    for column in score_columns[2:]:
        if column not in scores.columns:
            scores[column] = 0.0
    if scores.empty:
        for column in score_columns[2:]:
            out[column] = 0.0
    else:
        out = out.merge(scores[score_columns], on=["election_id", slot_column], how="left")
        for column in score_columns[2:]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    eligible_intensity = filter_available_by_election(
        intensity.copy(),
        election_dates,
        source_name="mega_issue_intensity_incumbent_shock",
    )
    eligible_intensity["mega_issue_intensity"] = pd.to_numeric(
        eligible_intensity.get("mega_issue_intensity", 1.0), errors="coerce"
    ).fillna(1.0).clip(lower=0.0)
    latest_intensity = (
        eligible_intensity.sort_values("available_date")
        .drop_duplicates("election_id", keep="last")
        .set_index("election_id")["mega_issue_intensity"]
    )
    out["mega_issue_intensity_response"] = (
        out["election_id"].astype(str).map(latest_intensity).fillna(1.0)
    )

    candidate_summary = out.groupby(["election_id", slot_column], as_index=False).agg(
        party_base=("direct_party_recent_base", "mean"),
        preliminary_share=(prediction_column, "mean"),
        party_reliability=("direct_party_reliability", "mean"),
    )
    candidate_summary["strongest_party_base"] = candidate_summary.groupby(
        "election_id"
    )["party_base"].transform("max")
    candidate_summary["party_base_resistance"] = (
        candidate_summary["party_base"]
        / candidate_summary["strongest_party_base"].replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    buffer = max(float(conversion_buffer), 1e-6)
    candidate_summary["conversion_resistance"] = (
        (candidate_summary["preliminary_share"] - candidate_summary["party_base"])
        / buffer
    ).clip(0.0, 1.0)
    candidate_summary["incumbent_burden_resistance"] = candidate_summary[
        ["party_base_resistance", "conversion_resistance"]
    ].max(axis=1)
    candidate_summary["incumbent_burden_exposure"] = (
        (1.0 - candidate_summary["incumbent_burden_resistance"])
        * candidate_summary["party_reliability"].clip(0.0, 1.0)
    )
    out = out.merge(
        candidate_summary[
            [
                "election_id",
                slot_column,
                "party_base_resistance",
                "conversion_resistance",
                "incumbent_burden_resistance",
                "incumbent_burden_exposure",
            ]
        ],
        on=["election_id", slot_column],
        how="left",
    )

    out["government_burden_log_shift"] = (
        float(max(government_burden_gain, 0.0))
        * out["government_direction_score"]
        * out["incumbent_burden_exposure"]
    )
    direct_score = pd.to_numeric(
        out.get("direct_mega_score", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    out["rupture_extra_log_shift"] = (
        float(max(rupture_extra_gain, 0.0))
        * direct_score
        * (out["mega_issue_intensity_response"] - 1.0).clip(lower=0.0)
    )
    out["incumbent_shock_log_shift"] = (
        out["government_burden_log_shift"] + out["rupture_extra_log_shift"]
    ).clip(-abs(float(log_shift_cap)), abs(float(log_shift_cap)))
    out["_incumbent_shock_raw"] = pd.to_numeric(
        out[prediction_column], errors="coerce"
    ).fillna(0.0).clip(lower=1e-12) * np.exp(out["incumbent_shock_log_shift"])
    denominator = out.groupby(["election_id", "region_id"])[
        "_incumbent_shock_raw"
    ].transform("sum")
    out[output_column] = out["_incumbent_shock_raw"] / denominator.replace(0.0, np.nan)
    return out.drop(columns="_incumbent_shock_raw")
