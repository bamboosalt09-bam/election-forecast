"""Compile direction/attribution issue inputs into forecast issue scores."""

from __future__ import annotations

from typing import Dict

import pandas as pd


ISSUE_SCORE_COLUMNS = [
    "date",
    "candidate_id",
    "issue_name",
    "salience_score",
    "direction_score",
    "candidate_link_score",
    "media_reliability_score",
    "final_issue_score",
    "available_date",
]


def compile_enhanced_issue_scores(
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Compile optional manual issue layers into ``issue_scores`` rows."""

    candidates = _candidate_frame(data.get("candidates", pd.DataFrame()))
    profiles = _prepare_profile_frame(
        data.get("candidate_issue_profile", pd.DataFrame()),
        candidates,
        forecast_date,
    )
    axis = _available(data.get("mega_issue_axis", pd.DataFrame()), forecast_date)
    attribution = _available(data.get("mega_issue_attribution", pd.DataFrame()), forecast_date)

    frames = [
        _profile_issue_scores(profiles),
        _mega_issue_scores(attribution, axis, candidates, profiles),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[ISSUE_SCORE_COLUMNS]


def merge_enhanced_issue_scores(
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return base issue scores plus optional enhanced issue rows."""

    base = data.get("issue_scores", pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)).copy()
    enhanced = compile_enhanced_issue_scores(data, forecast_date)
    frames = [frame for frame in [base, enhanced] if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[ISSUE_SCORE_COLUMNS]


def _available(frame: pd.DataFrame, forecast_date: str | pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "available_date" not in frame.columns:
        raise ValueError("enhanced issue input is missing available_date")
    out = frame.copy()
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    if out["available_date"].isna().any():
        raise ValueError("enhanced issue input contains missing or invalid available_date")
    cutoff = pd.Timestamp(forecast_date)
    return out.loc[out["available_date"].le(cutoff)].copy()


def _candidate_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = ["candidate_id", "candidate_name", "party_name", "official_camp"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    out = candidates.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str).str.strip()
    return out[columns].drop_duplicates()


def _prepare_profile_frame(
    profiles: pd.DataFrame,
    candidates: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
) -> pd.DataFrame:
    profiles = _available(profiles, forecast_date)
    if profiles.empty:
        return profiles
    out = profiles.copy()
    for column in ["candidate_id", "candidate_name", "slot", "election_id", "issue_name"]:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str).str.strip()

    if not candidates.empty:
        name_to_id = dict(zip(candidates["candidate_name"], candidates["candidate_id"]))
        missing_id = out["candidate_id"].eq("")
        out.loc[missing_id, "candidate_id"] = out.loc[missing_id, "candidate_name"].map(name_to_id).fillna("")
        out = out.loc[out["candidate_id"].isin(set(candidates["candidate_id"]))].copy()
    return out


def _profile_issue_scores(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)
    frame = profiles.loc[profiles["candidate_id"].fillna("").astype(str).str.strip().ne("")].copy()
    if frame.empty:
        return pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)
    for column in ["association_strength", "direction", "confidence"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    frame["salience_score"] = frame["association_strength"].clip(lower=0.0)
    frame["direction_score"] = frame["direction"].clip(lower=-1.0, upper=1.0)
    frame["candidate_link_score"] = 1.0
    frame["media_reliability_score"] = frame["confidence"].clip(lower=0.0, upper=1.0)
    frame["final_issue_score"] = (
        frame["salience_score"]
        * frame["direction_score"]
        * frame["candidate_link_score"]
        * frame["media_reliability_score"]
    ).clip(lower=-1.0, upper=1.0)
    return frame[ISSUE_SCORE_COLUMNS]


def _mega_issue_scores(
    attribution: pd.DataFrame,
    axis: pd.DataFrame,
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    if attribution.empty:
        return pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)
    joined = attribution.copy()
    if not axis.empty:
        joined = joined.merge(
            axis[
                [
                    "election_id",
                    "mega_event",
                    "primary_issue",
                    "secondary_issue",
                    "axis_weight",
                    "regime_axis_weight",
                ]
            ],
            on=["election_id", "mega_event"],
            how="left",
        )
    else:
        for column in ["primary_issue", "secondary_issue", "axis_weight", "regime_axis_weight"]:
            joined[column] = "" if column in {"primary_issue", "secondary_issue"} else 1.0

    rows: list[dict[str, object]] = []
    for record in joined.to_dict("records"):
        target_ids = _resolve_targets(record, candidates, profiles)
        if not target_ids:
            continue
        issue_name = _clean(record.get("issue_name")) or _clean(record.get("primary_issue"))
        if issue_name:
            rows.extend(
                _attribution_rows(
                    record,
                    target_ids,
                    issue_name,
                    _number(record.get("axis_weight"), 1.0),
                )
            )
        secondary = _clean(record.get("secondary_issue"))
        secondary_weight = _number(record.get("regime_axis_weight"), 0.0)
        if secondary and secondary != issue_name and secondary_weight > 0:
            rows.extend(_attribution_rows(record, target_ids, secondary, secondary_weight))

    if not rows:
        return pd.DataFrame(columns=ISSUE_SCORE_COLUMNS)
    return pd.DataFrame(rows, columns=ISSUE_SCORE_COLUMNS)


def _attribution_rows(
    record: dict[str, object],
    candidate_ids: list[str],
    issue_name: str,
    salience_score: float,
) -> list[dict[str, object]]:
    polarity = _number(record.get("polarity"), 0.0)
    weight = _number(record.get("weight"), 0.0)
    confidence = _number(record.get("confidence"), 0.0)
    available_date = pd.Timestamp(record.get("available_date"))
    out = []
    for candidate_id in candidate_ids:
        out.append(
            {
                "date": available_date,
                "candidate_id": candidate_id,
                "issue_name": issue_name,
                "salience_score": salience_score,
                "direction_score": max(-1.0, min(1.0, polarity)),
                "candidate_link_score": max(0.0, weight),
                "media_reliability_score": max(0.0, min(1.0, confidence)),
                "final_issue_score": max(-1.0, min(1.0, salience_score * polarity * weight * confidence)),
                "available_date": available_date,
            }
        )
    return out


def _resolve_targets(
    record: dict[str, object],
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
) -> list[str]:
    target_type = _clean(record.get("target_type"))
    target = _clean(record.get("target"))
    election_id = _clean(record.get("election_id"))
    if not target_type or not target:
        return []
    if target_type == "candidate_id":
        return [target]
    if target_type == "candidate_slot":
        if profiles.empty:
            return []
        rows = profiles.loc[
            (profiles["election_id"] == election_id)
            & (profiles["slot"] == target)
            & profiles["candidate_id"].fillna("").astype(str).str.strip().ne("")
        ]
        return sorted(rows["candidate_id"].dropna().astype(str).unique())
    if candidates.empty:
        return []
    if target_type == "party":
        rows = candidates.loc[candidates["party_name"] == target]
    elif target_type in {"camp", "incumbent_camp"}:
        rows = candidates.loc[candidates["official_camp"] == target]
    else:
        return []
    return sorted(rows["candidate_id"].dropna().astype(str).unique())


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _number(value: object, default: float) -> float:
    out = pd.to_numeric(value, errors="coerce")
    if pd.isna(out):
        return default
    return float(out)
