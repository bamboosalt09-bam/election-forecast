"""Bounded direct adjustment for unusually strong, explicitly attributed shocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.point_in_time import filter_available_by_election


POLITICAL_SHOCK_ISSUES = frozenset(
    {
        "corruption_integrity",
        "external_shock",
        "regime_change",
        "security_nk",
        "unification_event",
        "withdrawal_event",
    }
)
EVENT_CLASS_DIRECT_ISSUES = {
    "institutional_crisis": frozenset({"regime_change"}),
    "accountability_scandal": frozenset(
        {"corruption_integrity", "regime_change"}
    ),
    "incumbent_assessment": frozenset(
        {"corruption_integrity", "regime_change"}
    ),
    "political_realignment": frozenset({"external_shock", "regime_change"}),
}
DEFAULT_DIRECT_MEGA_GAIN = 0.40
DEFAULT_MINIMUM_INTENSITY = 1.0
DEFAULT_LOG_SHIFT_CAP = 0.20
DEFAULT_SCORE_CAP = 0.50


def align_profile_to_event_class(
    profile: pd.DataFrame,
    taxonomy: pd.DataFrame,
    election_dates: dict[str, str],
) -> pd.DataFrame:
    """Keep direct candidate shocks compatible with the election event class.

    Election-wide intensity describes one selected event environment.  Without
    this alignment an institutional-crisis intensity can accidentally amplify
    an unrelated candidate withdrawal or security mention merely because it
    has stronger attribution evidence.  Classes without a declared mapping are
    left unchanged for backward compatibility.
    """

    required = {"election_id", "issue_name"}
    taxonomy_required = {"election_id", "shock_type", "available_date"}
    if (
        profile.empty
        or taxonomy.empty
        or not required.issubset(profile.columns)
        or not taxonomy_required.issubset(taxonomy.columns)
    ):
        return profile.copy()
    eligible = filter_available_by_election(
        taxonomy.copy(), election_dates, source_name="direct_mega_event_taxonomy"
    )
    selected = (
        eligible.sort_values("available_date")
        .drop_duplicates("election_id", keep="last")
        .set_index("election_id")["shock_type"]
        .astype(str)
        .to_dict()
    )
    keep = []
    for row in profile[["election_id", "issue_name"]].itertuples(index=False):
        allowed = EVENT_CLASS_DIRECT_ISSUES.get(selected.get(str(row.election_id), ""))
        keep.append(allowed is None or str(row.issue_name) in allowed)
    return profile.loc[keep].copy()


def compile_direct_mega_scores(
    profile: pd.DataFrame,
    intensity: pd.DataFrame,
    election_dates: dict[str, str],
    *,
    minimum_intensity: float = DEFAULT_MINIMUM_INTENSITY,
    score_cap: float = DEFAULT_SCORE_CAP,
) -> pd.DataFrame:
    """Select one explicitly attributed high-intensity political shock per election."""

    columns = [
        "election_id",
        "slot",
        "issue_name",
        "direct_mega_score",
        "mega_issue_intensity",
        "selection_score",
    ]
    profile_required = {
        "election_id",
        "slot",
        "issue_name",
        "direction",
        "association_strength",
        "confidence",
        "target_absolute_evidence",
        "target_attribution_confidence",
        "available_date",
    }
    intensity_required = {
        "election_id",
        "mega_issue_intensity",
        "available_date",
    }
    if (
        profile.empty
        or intensity.empty
        or not profile_required.issubset(profile.columns)
        or not intensity_required.issubset(intensity.columns)
    ):
        return pd.DataFrame(columns=columns)

    eligible_profile = filter_available_by_election(
        profile.copy(), election_dates, source_name="candidate_issue_profile_direct_mega"
    )
    eligible_intensity = filter_available_by_election(
        intensity.copy(), election_dates, source_name="mega_issue_intensity_direct_mega"
    )
    eligible_intensity["mega_issue_intensity"] = pd.to_numeric(
        eligible_intensity["mega_issue_intensity"], errors="coerce"
    ).fillna(1.0).clip(lower=0.0)
    eligible_intensity = eligible_intensity.sort_values("available_date").drop_duplicates(
        "election_id", keep="last"
    )[["election_id", "mega_issue_intensity"]]

    frame = eligible_profile.merge(eligible_intensity, on="election_id", how="inner")
    for column in [
        "direction",
        "association_strength",
        "confidence",
        "target_absolute_evidence",
        "target_attribution_confidence",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.loc[
        frame["issue_name"].astype(str).isin(POLITICAL_SHOCK_ISSUES)
        & frame["direction"].abs().ge(0.10)
        & frame["target_absolute_evidence"].gt(0.0)
        & frame["mega_issue_intensity"].gt(float(minimum_intensity))
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["selection_score"] = (
        frame["target_attribution_confidence"].clip(0.0, 1.0)
        * np.log1p(frame["target_absolute_evidence"].clip(lower=0.0))
    )
    selected = (
        frame.sort_values(
            ["election_id", "selection_score", "issue_name"],
            ascending=[True, False, True],
        )
        .drop_duplicates("election_id")[["election_id", "issue_name"]]
    )
    frame = frame.merge(selected, on=["election_id", "issue_name"], how="inner")
    # Crossing the minimum must not switch on the full shock. Ramp the existing
    # intensity-scaled score from zero at the gate to full strength one
    # intensity unit above it. This preserves the legacy score at intensity 2
    # when the default gate is 1 while making continuous inputs stable.
    frame["intensity_activation"] = (
        frame["mega_issue_intensity"] - float(minimum_intensity)
    ).clip(0.0, 1.0)
    frame["direct_mega_score"] = (
        frame["direction"].clip(-1.0, 1.0)
        * frame["association_strength"].clip(0.0, 1.0)
        * frame["confidence"].clip(0.0, 1.0)
        * frame["mega_issue_intensity"].clip(0.0, 2.5)
        * frame["intensity_activation"]
    )
    grouped = frame.groupby(
        ["election_id", "slot", "issue_name", "mega_issue_intensity"],
        as_index=False,
    ).agg(
        direct_mega_score=("direct_mega_score", "sum"),
        selection_score=("selection_score", "max"),
    )
    grouped["direct_mega_score"] = grouped["direct_mega_score"].clip(
        -abs(float(score_cap)), abs(float(score_cap))
    )
    return grouped[columns]


def apply_direct_mega_shift(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    prediction_column: str,
    slot_column: str = "source_slot",
    output_column: str | None = None,
    gain: float = DEFAULT_DIRECT_MEGA_GAIN,
    log_shift_cap: float = DEFAULT_LOG_SHIFT_CAP,
) -> pd.DataFrame:
    """Apply a bounded log-share shift and renormalize each regional contest."""

    output_column = output_column or prediction_column
    required = {"election_id", "region_id", slot_column, prediction_column}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"direct mega shift frame missing columns: {missing}")
    out = frame.copy()
    if scores.empty:
        out["direct_mega_score"] = 0.0
        out["direct_mega_issue"] = ""
    else:
        score_frame = scores.rename(
            columns={"slot": slot_column, "issue_name": "direct_mega_issue"}
        )[["election_id", slot_column, "direct_mega_issue", "direct_mega_score"]]
        out = out.merge(
            score_frame,
            on=["election_id", slot_column],
            how="left",
            validate="many_to_one",
        )
        out["direct_mega_score"] = pd.to_numeric(
            out["direct_mega_score"], errors="coerce"
        ).fillna(0.0)
        out["direct_mega_issue"] = out["direct_mega_issue"].fillna("")
    cap = abs(float(log_shift_cap))
    out["direct_mega_log_shift"] = (
        out["direct_mega_score"] * float(gain)
    ).clip(-cap, cap)
    raw = np.clip(pd.to_numeric(out[prediction_column], errors="coerce"), 1e-8, 1.0)
    out["_direct_mega_raw"] = raw * np.exp(out["direct_mega_log_shift"])
    denominator = out.groupby(["election_id", "region_id"])["_direct_mega_raw"].transform("sum")
    out[output_column] = out["_direct_mega_raw"] / denominator.replace(0.0, np.nan)
    return out.drop(columns="_direct_mega_raw")
