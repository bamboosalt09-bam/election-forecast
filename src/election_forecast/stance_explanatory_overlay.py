"""Compile ambiguity-gated Assembly stance into a bounded issue overlay.

The classifier is explanatory rather than a direct vote-share model. Neutral
sentences retain unsigned information mass, while only ambiguity-gated
directional rows affect issue character and candidate-link reliability.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from election_forecast.stance_v3 import classify_issue_character


LABEL_SIGN = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
REQUIRED_COLUMNS = {
    "election_id",
    "meeting_date",
    "issue_name",
    "speaker",
    "committee",
    "target_type",
    "target_name",
    "text_sha256",
    "context_confidence",
    "ambiguity_gated_prediction",
}


def _bounded_log(value: float, cap: float) -> float:
    if value <= 0.0:
        return 0.0
    return float(min(np.log1p(value) / np.log1p(cap), 1.0))


def _candidate_maps(candidates: pd.DataFrame) -> dict[str, dict[tuple[str, str], str]]:
    required = {"election_id", "slot", "candidate_name", "party_name"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate reference missing columns: {sorted(missing)}")
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


def _government_responsibility_map(
    responsibility: pd.DataFrame | None,
) -> dict[str, tuple[str, pd.Timestamp | pd.NaT]]:
    """Map a government target to the pre-election incumbent-continuity slot."""

    if responsibility is None or responsibility.empty:
        return {}
    required = {"election_id", "slot", "responsibility_score", "available_date"}
    missing = required - set(responsibility.columns)
    if missing:
        raise ValueError(f"responsibility metadata missing columns: {sorted(missing)}")
    frame = responsibility.copy()
    frame["responsibility_score"] = pd.to_numeric(
        frame["responsibility_score"], errors="coerce"
    ).fillna(0.0)
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    frame = frame.loc[frame["responsibility_score"].gt(0.0)].copy()
    if frame.empty:
        return {}
    selected = (
        frame.sort_values(
            ["election_id", "responsibility_score", "slot"],
            ascending=[True, False, True],
        )
        .drop_duplicates("election_id", keep="first")
    )
    return {
        str(row.election_id): (str(row.slot), row.available_date)
        for row in selected.itertuples(index=False)
    }


def _incumbent_person_map(
    incumbent_targets: pd.DataFrame | None,
    government_map: dict[str, tuple[str, pd.Timestamp | pd.NaT]],
) -> dict[tuple[str, str], tuple[str, pd.Timestamp | pd.NaT]]:
    if incumbent_targets is None or incumbent_targets.empty:
        return {}
    required = {"election_id", "target_name", "available_date"}
    missing = required - set(incumbent_targets.columns)
    if missing:
        raise ValueError(f"incumbent target metadata missing columns: {sorted(missing)}")
    out: dict[tuple[str, str], tuple[str, pd.Timestamp | pd.NaT]] = {}
    for row in incumbent_targets.itertuples(index=False):
        election_id = str(row.election_id)
        slot, responsibility_date = government_map.get(election_id, ("", pd.NaT))
        if not slot:
            continue
        alias_date = pd.to_datetime(row.available_date, errors="coerce")
        dates = pd.Series([responsibility_date, alias_date]).dropna()
        available_date = dates.max() if not dates.empty else pd.NaT
        out[(election_id, str(row.target_name).strip())] = (slot, available_date)
    return out


def _speaker_name(value: object) -> str:
    text = str(value).strip()
    match = re.match(r"^([가-힣]{2,4})\s*(?:의원|위원)$", text)
    return match.group(1) if match else ""


def _speaker_bloc_map(member_history: pd.DataFrame | None) -> dict[tuple[str, str], str]:
    if member_history is None or member_history.empty:
        return {}
    required = {"daesu", "name", "bloc"}
    missing = required - set(member_history.columns)
    if missing:
        raise ValueError(f"member history missing columns: {sorted(missing)}")
    return {
        (str(row.daesu).strip(), str(row.name).strip()): str(row.bloc).strip()
        for row in member_history.itertuples(index=False)
        if str(row.name).strip() and str(row.bloc).strip()
    }


def _global_issue_rows(frame: pd.DataFrame, character_gain: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    unique = frame.drop_duplicates(["election_id", "issue_name", "text_sha256"])
    for (election_id, issue_name), group in unique.groupby(
        ["election_id", "issue_name"], sort=True
    ):
        confidence = pd.to_numeric(group["context_confidence"], errors="coerce").fillna(0.0)
        confidence = confidence.clip(0.0, 1.0)
        quality = float(np.clip((confidence.mean() - 1.0 / 3.0) / (2.0 / 3.0), 0.0, 1.0))
        labels = group["ambiguity_gated_prediction"].astype(str)
        masses = {
            label: float(confidence.loc[labels.eq(label)].sum())
            for label in LABEL_SIGN
        }
        total_mass = sum(masses.values())
        if total_mass <= 0.0:
            shares = {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        else:
            shares = {label: mass / total_mass for label, mass in masses.items()}
        character = classify_issue_character(
            shares["negative"],
            shares["neutral"],
            shares["positive"],
            confidence_quality=quality,
        )
        evidence_count = int(len(group))
        speaker_count = int(group["speaker"].astype(str).nunique())
        committee_count = int(group["committee"].astype(str).nunique())
        raw_strength = (
            0.60 * _bounded_log(evidence_count, 500.0)
            + 0.25 * _bounded_log(speaker_count, 100.0)
            + 0.10 * _bounded_log(committee_count, 12.0)
            + 0.05 * quality
        )
        rows.append(
            {
                "election_id": str(election_id),
                "issue_name": str(issue_name),
                "issue_evidence_count": evidence_count,
                "issue_speaker_count": speaker_count,
                "issue_committee_count": committee_count,
                "issue_directional_count": int(labels.ne("neutral").sum()),
                "issue_confidence_quality": quality,
                "issue_raw_strength": raw_strength,
                "negative_share": shares["negative"],
                "neutral_share": shares["neutral"],
                "positive_share": shares["positive"],
                **character,
                "issue_available_date": pd.to_datetime(
                    group["meeting_date"], errors="coerce"
                ).max(),
            }
        )
    issues = pd.DataFrame(rows)
    if issues.empty:
        return issues
    issues["issue_percentile"] = issues.groupby("election_id")["issue_raw_strength"].rank(
        method="average", pct=True
    )
    centered_attention = 2.0 * (issues["issue_percentile"] - 0.5)
    issues["attention_multiplier"] = (
        1.0 + 0.04 * centered_attention * (0.5 + 0.5 * issues["issue_confidence_quality"])
    ).clip(0.96, 1.04)
    raw_character = (
        1.0
        + float(character_gain)
        * issues["issue_confidence_quality"]
        * issues["character_score"]
    ).clip(0.95, 1.05)
    election_mean = raw_character.groupby(issues["election_id"]).transform("mean")
    issues["character_multiplier_raw"] = raw_character
    issues["character_multiplier"] = (raw_character / election_mean).clip(0.95, 1.05)
    issues["salience_multiplier"] = issues["character_multiplier"]
    issues["character_gain"] = float(character_gain)
    return issues


def compile_explanatory_overlay(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    character_gain: float = 0.04,
    link_gain: float = 0.01,
    government_responsibility: pd.DataFrame | None = None,
    incumbent_targets: pd.DataFrame | None = None,
    member_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one PIT-dated issue overlay row per election, issue, and slot."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"stance application missing columns: {sorted(missing)}")
    if not 0.0 <= character_gain <= 0.10:
        raise ValueError("character_gain must be between 0 and 0.10")
    if not 0.0 <= link_gain <= 0.02:
        raise ValueError("link_gain must be between 0 and 0.02")
    labels = set(frame["ambiguity_gated_prediction"].astype(str).unique())
    unknown = labels - set(LABEL_SIGN)
    if unknown:
        raise ValueError(f"unsupported gated labels: {sorted(unknown)}")

    issues = _global_issue_rows(frame, character_gain)
    slots = candidates[["election_id", "slot"]].drop_duplicates().copy()
    slots["election_id"] = slots["election_id"].astype(str)
    overlay = issues.merge(slots, on="election_id", how="inner", validate="many_to_many")

    maps = _candidate_maps(candidates)
    government_map = _government_responsibility_map(government_responsibility)
    incumbent_map = _incumbent_person_map(incumbent_targets, government_map)
    slot_bloc = {
        (str(row.election_id), str(row.slot)): str(
            getattr(row, "candidate_bloc", getattr(row, "party_name", ""))
        ).strip()
        for row in candidates.itertuples(index=False)
    }
    speaker_bloc = _speaker_bloc_map(member_history)
    target = frame.loc[
        frame["target_type"].isin(["person", "party", "government"])
        & frame["ambiguity_gated_prediction"].ne("neutral")
    ].copy()
    mapped = [
        government_map.get(str(election_id), ("", pd.NaT))[0]
        if str(target_type) == "government"
        else (
            maps["person"].get((str(election_id), str(target_name)), "")
            or incumbent_map.get((str(election_id), str(target_name).strip()), ("", pd.NaT))[0]
            if str(target_type) == "person"
            else maps["party"].get((str(election_id), str(target_name)), "")
        )
        for election_id, target_type, target_name in zip(
            target["election_id"],
            target["target_type"],
            target["target_name"],
            strict=True,
        )
    ]
    target["slot"] = mapped
    target["responsibility_available_date"] = [
        government_map.get(str(election_id), ("", pd.NaT))[1]
        if str(target_type) == "government"
        else (
            incumbent_map.get((str(election_id), str(target_name).strip()), ("", pd.NaT))[1]
            if str(target_type) == "person"
            else pd.NaT
        )
        for election_id, target_type, target_name in zip(
            target["election_id"],
            target["target_type"],
            target["target_name"],
            strict=True,
        )
    ]
    target["target_bloc"] = [
        slot_bloc.get((str(election_id), str(slot)), "")
        for election_id, slot in zip(target["election_id"], target["slot"], strict=True)
    ]
    if "assembly_daesu" in target.columns:
        target["speaker_name_normalized"] = target["speaker"].map(_speaker_name)
        target["speaker_bloc"] = [
            speaker_bloc.get((str(daesu).strip(), str(name).strip()), "")
            for daesu, name in zip(
                target["assembly_daesu"], target["speaker_name_normalized"], strict=True
            )
        ]
    else:
        target["speaker_bloc"] = ""
    target = target.loc[target["slot"].ne("")].drop_duplicates(
        ["election_id", "issue_name", "slot", "text_sha256"]
    )
    link_rows: list[dict[str, object]] = []
    for (election_id, issue_name, slot), group in target.groupby(
        ["election_id", "issue_name", "slot"], sort=True
    ):
        confidence = pd.to_numeric(group["context_confidence"], errors="coerce").fillna(0.0)
        signs = group["ambiguity_gated_prediction"].map(LABEL_SIGN).astype(float)
        same_bloc = group["speaker_bloc"].astype(str).ne("") & group[
            "speaker_bloc"
        ].astype(str).eq(group["target_bloc"].astype(str))
        cross_bloc = group["speaker_bloc"].astype(str).ne("") & group[
            "target_bloc"
        ].astype(str).ne("") & ~same_bloc
        relation_weight = pd.Series(0.85, index=group.index, dtype=float)
        relation_weight.loc[same_bloc & signs.lt(0.0)] = 1.15
        relation_weight.loc[same_bloc & signs.gt(0.0)] = 0.85
        relation_weight.loc[cross_bloc & signs.lt(0.0)] = 0.85
        relation_weight.loc[cross_bloc & signs.gt(0.0)] = 1.10
        weighted_confidence = confidence * relation_weight
        signed = signs * weighted_confidence
        mass = float(signed.abs().sum())
        signed_mass = float(signed.sum())
        consistency = abs(float(signed.sum())) / mass if mass > 0.0 else 0.0
        reliability = (
            0.65 * consistency + 0.35 * _bounded_log(len(group), 20.0)
        ) * float(weighted_confidence.mean())
        negative_mass = float(weighted_confidence.loc[signs.lt(0.0)].sum())
        positive_mass = float(weighted_confidence.loc[signs.gt(0.0)].sum())
        responsibility_dates = pd.to_datetime(
            group["responsibility_available_date"], errors="coerce"
        )
        link_rows.append(
            {
                "election_id": str(election_id),
                "issue_name": str(issue_name),
                "slot": str(slot),
                "link_evidence_count": int(len(group)),
                "link_consistency": consistency,
                "link_reliability": reliability,
                "target_signed_evidence": signed_mass,
                "target_absolute_evidence": mass,
                "target_directional_balance": signed_mass / mass if mass > 0.0 else 0.0,
                "target_negative_mass": negative_mass,
                "target_positive_mass": positive_mass,
                "target_attribution_confidence": reliability,
                "same_bloc_target_evidence_count": int(same_bloc.sum()),
                "cross_bloc_target_evidence_count": int(cross_bloc.sum()),
                "speaker_bloc_known_count": int(group["speaker_bloc"].astype(str).ne("").sum()),
                "target_source_types": "|".join(
                    sorted(set(group["target_type"].astype(str)))
                ),
                "link_available_date": pd.to_datetime(
                    group["meeting_date"], errors="coerce"
                ).max(),
                "responsibility_available_date": responsibility_dates.max(),
            }
        )
    links = pd.DataFrame(link_rows)
    if not links.empty:
        overlay = overlay.merge(
            links, on=["election_id", "issue_name", "slot"], how="left"
        )
    numeric_link_columns = (
        "link_evidence_count",
        "link_consistency",
        "link_reliability",
        "target_signed_evidence",
        "target_absolute_evidence",
        "target_directional_balance",
        "target_negative_mass",
        "target_positive_mass",
        "target_attribution_confidence",
        "same_bloc_target_evidence_count",
        "cross_bloc_target_evidence_count",
        "speaker_bloc_known_count",
    )
    for column in numeric_link_columns:
        overlay[column] = pd.to_numeric(overlay.get(column, 0.0), errors="coerce").fillna(0.0)
    overlay["link_evidence_count"] = overlay["link_evidence_count"].astype(int)
    overlay["target_source_types"] = overlay.get("target_source_types", "").fillna("")
    overlay["link_multiplier_raw"] = 1.0 + float(link_gain) * overlay["link_reliability"]
    link_mean = overlay.groupby(["election_id", "issue_name"])["link_multiplier_raw"].transform(
        "mean"
    )
    overlay["link_multiplier"] = (overlay["link_multiplier_raw"] / link_mean).clip(0.98, 1.02)
    issue_dates = pd.to_datetime(overlay["issue_available_date"], errors="coerce")
    link_dates = pd.to_datetime(overlay.get("link_available_date"), errors="coerce")
    responsibility_dates = pd.to_datetime(
        overlay.get("responsibility_available_date"), errors="coerce"
    )
    overlay["available_date"] = pd.concat(
        [issue_dates, link_dates, responsibility_dates], axis=1
    ).max(axis=1)
    overlay["available_date"] = overlay["available_date"].dt.strftime("%Y-%m-%d")
    overlay["source_model"] = "stance_nli_ambiguity_v14"
    overlay["link_gain"] = float(link_gain)
    return overlay.sort_values(["election_id", "issue_name", "slot"]).reset_index(drop=True)


def required_output_columns() -> Iterable[str]:
    return (
        "election_id",
        "issue_name",
        "slot",
        "salience_multiplier",
        "link_multiplier",
        "available_date",
    )
