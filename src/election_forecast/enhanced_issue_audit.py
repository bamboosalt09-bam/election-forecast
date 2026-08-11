"""Audit optional enhanced issue seed CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from election_forecast.enhanced_issues import compile_enhanced_issue_scores


@dataclass(frozen=True)
class AuditFinding:
    """One enhanced issue seed audit finding."""

    severity: str
    check: str
    message: str
    rows: str = ""


def audit_enhanced_issue_inputs(
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
    target_election_id: str | None = None,
    aggregate_abs_threshold: float = 2.0,
) -> pd.DataFrame:
    """Return warnings/errors for enhanced issue seed configuration."""

    findings: list[AuditFinding] = []
    profile = _target(data.get("candidate_issue_profile", pd.DataFrame()), target_election_id)
    axis = _target(data.get("mega_issue_axis", pd.DataFrame()), target_election_id)
    attribution = _target(data.get("mega_issue_attribution", pd.DataFrame()), target_election_id)
    scope = data.get("issue_scope_weights", pd.DataFrame())
    candidates = _target(data.get("candidates", pd.DataFrame()), target_election_id)

    _audit_issue_scope(findings, profile, axis, attribution, scope)
    _audit_axis_attribution_refs(findings, axis, attribution)
    _audit_slot_refs(findings, profile, attribution)
    _audit_candidate_activation(findings, profile, candidates, target_election_id)
    _audit_compiled_scores(findings, data, forecast_date, target_election_id, aggregate_abs_threshold)

    return pd.DataFrame([finding.__dict__ for finding in findings], columns=["severity", "check", "message", "rows"])


def _target(frame: pd.DataFrame, target_election_id: str | None) -> pd.DataFrame:
    if frame.empty or not target_election_id or "election_id" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["election_id"].astype(str) == str(target_election_id)].copy()


def _audit_issue_scope(
    findings: list[AuditFinding],
    profile: pd.DataFrame,
    axis: pd.DataFrame,
    attribution: pd.DataFrame,
    scope: pd.DataFrame,
) -> None:
    if scope.empty:
        findings.append(AuditFinding("error", "scope", "issue_scope_weights.csv is empty"))
        return
    scoped = set(scope["issue_name"].dropna().astype(str))
    used: set[str] = set()
    if not profile.empty:
        used.update(profile["issue_name"].dropna().astype(str))
    if not attribution.empty:
        used.update(attribution["issue_name"].dropna().astype(str))
    if not axis.empty:
        used.update(axis["primary_issue"].dropna().astype(str))
        used.update(axis["secondary_issue"].dropna().astype(str))
    used.discard("")
    missing = sorted(used - scoped)
    if missing:
        findings.append(AuditFinding("error", "scope", "issues missing scope weights", "|".join(missing)))


def _audit_axis_attribution_refs(
    findings: list[AuditFinding],
    axis: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    if attribution.empty:
        return
    axis_keys = set(zip(axis.get("election_id", []), axis.get("mega_event", [])))
    missing = []
    for idx, row in attribution.iterrows():
        if (row.get("election_id"), row.get("mega_event")) not in axis_keys:
            missing.append(str(idx))
    if missing:
        findings.append(
            AuditFinding("error", "mega_axis_ref", "mega attribution rows without matching mega axis", "|".join(missing))
        )


def _audit_slot_refs(
    findings: list[AuditFinding],
    profile: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    if profile.empty or attribution.empty:
        return
    profile_slots = set(zip(profile["election_id"].astype(str), profile["slot"].fillna("").astype(str)))
    missing = []
    rows = attribution.loc[attribution["target_type"].astype(str) == "candidate_slot"]
    for idx, row in rows.iterrows():
        if (str(row.get("election_id")), str(row.get("target"))) not in profile_slots:
            missing.append(str(idx))
    if missing:
        findings.append(
            AuditFinding("error", "candidate_slot_ref", "candidate_slot attribution rows without profile slot", "|".join(missing))
        )


def _audit_candidate_activation(
    findings: list[AuditFinding],
    profile: pd.DataFrame,
    candidates: pd.DataFrame,
    target_election_id: str | None,
) -> None:
    if profile.empty or candidates.empty:
        return
    candidate_ids = set(candidates["candidate_id"].dropna().astype(str))
    inactive = sorted(set(profile["candidate_id"].dropna().astype(str)) - candidate_ids)
    if inactive and target_election_id:
        findings.append(
            AuditFinding(
                "warning",
                "inactive_profile",
                "profile rows for target election do not match active candidates",
                "|".join(inactive),
            )
        )


def _audit_compiled_scores(
    findings: list[AuditFinding],
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
    target_election_id: str | None,
    aggregate_abs_threshold: float,
) -> None:
    filtered = dict(data)
    if target_election_id:
        for name in ["candidates", "candidate_issue_profile", "mega_issue_axis", "mega_issue_attribution"]:
            frame = filtered.get(name)
            if frame is not None and not frame.empty and "election_id" in frame.columns:
                filtered[name] = frame.loc[frame["election_id"].astype(str) == str(target_election_id)].copy()
    profile = filtered.get("candidate_issue_profile", pd.DataFrame())
    candidates = filtered.get("candidates", pd.DataFrame())
    if not profile.empty and (
        candidates.empty
        or not set(profile["candidate_id"].dropna().astype(str)).issubset(
            set(candidates.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
        )
    ):
        filtered["candidates"] = _seed_candidates(profile, candidates)
    compiled = compile_enhanced_issue_scores(filtered, forecast_date)
    if compiled.empty:
        if target_election_id:
            findings.append(
                AuditFinding("warning", "compiled_scores", "no enhanced issue scores compile for target election")
            )
        return
    outside = compiled.loc[~compiled["final_issue_score"].between(-1.0, 1.0)]
    if not outside.empty:
        findings.append(
            AuditFinding("error", "compiled_scores", "compiled final_issue_score outside [-1, 1]", _row_ids(outside))
        )
    aggregate = compiled.groupby(["candidate_id", "issue_name"], as_index=False)["final_issue_score"].sum()
    large = aggregate.loc[aggregate["final_issue_score"].abs().gt(aggregate_abs_threshold)]
    if not large.empty:
        labels = [
            f"{row.candidate_id}:{row.issue_name}:{row.final_issue_score:.3f}"
            for row in large.itertuples(index=False)
        ]
        findings.append(
            AuditFinding(
                "warning",
                "aggregate_signal",
                f"candidate issue aggregate exceeds abs threshold {aggregate_abs_threshold}",
                "|".join(labels),
            )
        )


def _row_ids(frame: pd.DataFrame) -> str:
    return "|".join(str(idx) for idx in frame.index)


def _seed_candidates(profile: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["candidate_id", "candidate_name", "party_name", "official_camp"]
    if candidates.empty:
        existing = pd.DataFrame(columns=base_columns)
    else:
        existing = candidates.copy()
        for column in base_columns:
            if column not in existing.columns:
                existing[column] = ""
        existing = existing[base_columns]
    profile_candidates = profile[["candidate_id", "candidate_name"]].dropna(subset=["candidate_id"]).copy()
    profile_candidates["party_name"] = ""
    profile_candidates["official_camp"] = ""
    combined = pd.concat([existing, profile_candidates[base_columns]], ignore_index=True)
    combined["candidate_id"] = combined["candidate_id"].fillna("").astype(str)
    combined = combined.loc[combined["candidate_id"].ne("")]
    return combined.drop_duplicates(subset=["candidate_id"], keep="first")
