"""Build same-party versus cross-party treatment proxies for candidates.

This script uses existing assembly issue phrase matches. It does not reprocess
raw transcripts and does not use polling or news. The resulting features are
candidate-level proxies for whether the candidate's own bloc emphasized issues
that are favorable to the candidate, and whether other blocs emphasized issues
that are unfavorable to the candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from presidential_issue_engine.point_in_time import cutoff_dates_as_strings, filter_available_by_election  # noqa: E402
from presidential_issue_engine.election_scope import ELECTION_DATES  # noqa: E402

from assembly_bloc_issue_posture import (  # noqa: E402
    diagnostics_for,
    load_assembly_bloc_issue_posture,
    prior_election_rows,
    vector_for,
)
from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402


RESULTS = ROOT / "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
CANDIDATE_ISSUES = ROOT / "data/candidate_issue_link.csv"
CANDIDATE_PROFILE = ROOT / "data/raw/candidate_issue_profile.csv"
MEGA_ATTRIBUTION = ROOT / "data/raw/mega_issue_attribution.csv"
OUTPUT = ROOT / "data/raw/candidate_party_tone_gap.csv"

RISK_ISSUES = {"corruption_integrity", "family_legal_risk", "gaffe_event"}
LEGITIMACY_ISSUES = {"candidate_competence", "economy_growth", "welfare_pension"}
ALTERNATIVE_ISSUES = {"regime_change", "unification_event", "endorsement_event"}


def _safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


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
    out["candidate_bloc"] = out["party_name"].map(normalize_bloc)
    return out


def _candidate_issue_vectors() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    issues = filter_available_by_election(
        _safe_read(CANDIDATE_ISSUES),
        ELECTION_DATES,
        source_name="candidate_issue_link",
    )
    if not issues.empty:
        issues = issues[["election_id", "slot", "issue_name", "emphasis_within"]].copy()
        issues["candidate_issue_weight"] = pd.to_numeric(
            issues["emphasis_within"],
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        pieces.append(issues[["election_id", "slot", "issue_name", "candidate_issue_weight"]])
    profile = _safe_read(CANDIDATE_PROFILE)
    required = {"election_id", "slot", "issue_name", "association_strength", "confidence"}
    if not profile.empty and required.issubset(profile.columns):
        profile = profile[["election_id", "slot", "issue_name", "association_strength", "confidence"]].copy()
        profile["candidate_issue_weight"] = (
            pd.to_numeric(profile["association_strength"], errors="coerce").fillna(0.0).abs()
            * pd.to_numeric(profile["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
        pieces.append(profile[["election_id", "slot", "issue_name", "candidate_issue_weight"]])
    if not pieces:
        return pd.DataFrame(columns=["election_id", "slot", "issue_name", "candidate_issue_weight"])
    issues = pd.concat(pieces, ignore_index=True)
    issues = (
        issues.groupby(["election_id", "slot", "issue_name"], as_index=False)["candidate_issue_weight"]
        .sum()
    )
    totals = issues.groupby(["election_id", "slot"])["candidate_issue_weight"].transform("sum")
    issues["candidate_issue_weight"] = np.where(
        totals.to_numpy(float) > 0.0,
        issues["candidate_issue_weight"].to_numpy(float) / totals.to_numpy(float),
        0.0,
    )
    return issues[["election_id", "slot", "issue_name", "candidate_issue_weight"]]


def _fallback_issue_valence(issue_name: str) -> float:
    if issue_name in RISK_ISSUES:
        return -0.45
    if issue_name in LEGITIMACY_ISSUES:
        return 0.30
    if issue_name in ALTERNATIVE_ISSUES:
        return 0.25
    return 0.0


def _manual_issue_valence() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    profile = _safe_read(CANDIDATE_PROFILE)
    if not profile.empty:
        for row in profile.itertuples(index=False):
            rows.append(
                {
                    "election_id": getattr(row, "election_id", ""),
                    "slot": getattr(row, "slot", ""),
                    "issue_name": getattr(row, "issue_name", ""),
                    "valence_sum": float(getattr(row, "direction", 0.0))
                    * float(getattr(row, "association_strength", 0.0))
                    * float(getattr(row, "confidence", 0.0)),
                    "valence_weight": abs(float(getattr(row, "association_strength", 0.0)))
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
                    "issue_name": getattr(row, "issue_name", ""),
                    "valence_sum": float(getattr(row, "polarity", 0.0))
                    * float(getattr(row, "weight", 0.0))
                    * float(getattr(row, "confidence", 0.0)),
                    "valence_weight": abs(float(getattr(row, "weight", 0.0)))
                    * float(getattr(row, "confidence", 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["election_id", "slot", "issue_name", "issue_valence", "manual_valence_strength"])
    out = pd.DataFrame(rows)
    out = out.groupby(["election_id", "slot", "issue_name"], as_index=False).sum()
    out["issue_valence"] = np.where(
        out["valence_weight"].to_numpy(float) > 0.0,
        out["valence_sum"].to_numpy(float) / out["valence_weight"].to_numpy(float),
        0.0,
    )
    out["issue_valence"] = out["issue_valence"].clip(-1.0, 1.0)
    out["manual_valence_strength"] = out["valence_weight"].clip(0.0, 1.0)
    return out[["election_id", "slot", "issue_name", "issue_valence", "manual_valence_strength"]]


def _bloc_issue_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_assembly_bloc_issue_posture()


def build_tone_gap() -> pd.DataFrame:
    base = _candidate_base()
    cand_issues = _candidate_issue_vectors()
    manual_valence = _manual_issue_valence()
    bloc_issues, bloc_diag = _bloc_issue_profiles()
    if base.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for candidate in base.itertuples(index=False):
        election_id = str(candidate.election_id)
        slot = str(candidate.slot)
        candidate_bloc = str(candidate.candidate_bloc)
        issues = cand_issues.loc[
            (cand_issues["election_id"].astype(str) == election_id)
            & (cand_issues["slot"].astype(str) == slot)
        ].copy()
        if issues.empty:
            rows.append(_empty_row(candidate))
            continue
        vals = manual_valence.loc[
            (manual_valence["election_id"].astype(str) == election_id)
            & (manual_valence["slot"].astype(str) == slot)
        ]
        issues = issues.merge(vals, on=["election_id", "slot", "issue_name"], how="left")
        issues["manual_valence_strength"] = pd.to_numeric(
            issues["manual_valence_strength"],
            errors="coerce",
        ).fillna(0.0)
        fallback = issues["issue_name"].map(_fallback_issue_valence)
        issues["issue_valence"] = pd.to_numeric(issues["issue_valence"], errors="coerce")
        issues["issue_valence"] = issues["issue_valence"].fillna(fallback).clip(-1.0, 1.0)

        same = (
            vector_for(bloc_issues, election_id, candidate_bloc)
            .rename("same_bloc_issue_weight")
        )
        same.index.name = "issue_name"
        same = same.reset_index()
        other = (
            bloc_issues.loc[
                bloc_issues["election_id"].astype(str).eq(election_id)
                & (bloc_issues["bloc"].astype(str) != candidate_bloc)
            ]
            .groupby("issue_name", as_index=False)["bloc_issue_raw_weight"]
            .sum()
        )
        if other.empty:
            prior_issues = prior_election_rows(bloc_issues, election_id)
            other = (
                prior_issues.loc[prior_issues["bloc"].astype(str) != candidate_bloc]
                .groupby("issue_name", as_index=False)["bloc_issue_raw_weight"]
                .sum()
            )
        if not other.empty:
            total = float(other["bloc_issue_raw_weight"].sum())
            other["other_bloc_issue_weight"] = (
                other["bloc_issue_raw_weight"] / total if total > 0.0 else 0.0
            )
            other = other[["issue_name", "other_bloc_issue_weight"]]
        else:
            other = pd.DataFrame(columns=["issue_name", "other_bloc_issue_weight"])
        issues = issues.merge(same, on="issue_name", how="left").merge(other, on="issue_name", how="left")
        for column in ["same_bloc_issue_weight", "other_bloc_issue_weight"]:
            issues[column] = pd.to_numeric(issues[column], errors="coerce").fillna(0.0).clip(lower=0.0)

        positive = issues["issue_valence"].clip(lower=0.0)
        negative = (-issues["issue_valence"]).clip(lower=0.0)
        candidate_weight = issues["candidate_issue_weight"].clip(lower=0.0)
        same_joint = candidate_weight * issues["same_bloc_issue_weight"]
        other_joint = candidate_weight * issues["other_bloc_issue_weight"]

        same_positive = float((same_joint * positive).sum())
        same_negative = float((same_joint * negative).sum())
        other_positive = float((other_joint * positive).sum())
        other_negative = float((other_joint * negative).sum())
        same_net = same_positive - same_negative
        other_net = other_positive - other_negative
        same_supportive = max(0.0, same_net)
        cross_adverse = max(0.0, -other_net)
        contrast = same_net - other_net
        # These are stance proxies, not sentence-level sentiment labels.  They
        # combine candidate valence with the issue emphasis of own/other blocs.
        same_endorsement = same_positive
        same_defense = same_negative
        cross_attack = other_negative
        cross_rebuttal = other_positive
        party_stance_signal = same_endorsement + 0.50 * same_defense - 0.50 * cross_attack
        manual_coverage = float((candidate_weight * issues["manual_valence_strength"].clip(0.0, 1.0)).sum())
        same_diag = diagnostics_for(bloc_diag, election_id, candidate_bloc)
        same_speakers = float(same_diag.get("bloc_unique_speakers", 0.0)) if not same_diag.empty else 0.0
        other_diag = bloc_diag.loc[
            bloc_diag["election_id"].astype(str).eq(election_id)
            & (bloc_diag["bloc"].astype(str) != candidate_bloc)
        ] if not bloc_diag.empty else pd.DataFrame()
        if other_diag.empty and not bloc_diag.empty:
            prior_diag = prior_election_rows(bloc_diag, election_id)
            other_diag = prior_diag.loc[prior_diag["bloc"].astype(str) != candidate_bloc]
        other_speakers = float(
            other_diag["bloc_unique_speakers"].sum()
        ) if not bloc_diag.empty else 0.0
        speaker_coverage = min(1.0, (same_speakers + 0.5 * other_speakers) / 80.0)
        confidence = float(np.clip(0.35 + 0.35 * speaker_coverage + 0.30 * manual_coverage, 0.0, 0.85))
        rows.append(
            {
                "election_id": election_id,
                "slot": slot,
                "candidate_name": candidate.candidate_name,
                "candidate_bloc": candidate_bloc,
                "same_party_positive_tone": same_positive,
                "same_party_negative_tone": same_negative,
                "same_party_net_tone": same_net,
                "same_party_supportive_tone": same_supportive,
                "cross_party_positive_tone": other_positive,
                "cross_party_negative_tone": other_negative,
                "cross_party_net_tone": other_net,
                "cross_party_adverse_tone": cross_adverse,
                "party_tone_contrast": contrast,
                "same_party_endorsement_proxy": same_endorsement,
                "same_party_defense_proxy": same_defense,
                "cross_party_attack_proxy": cross_attack,
                "cross_party_rebuttal_proxy": cross_rebuttal,
                "party_stance_signal": party_stance_signal,
                "manual_valence_coverage": manual_coverage,
                "available_date": cutoff_dates_as_strings(ELECTION_DATES).get(election_id, ""),
                "confidence": confidence,
                "notes": "15th-22nd Assembly speaker/member-history issue posture proxy; no raw transcript reprocessing, news, or polling",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    nonnegative_columns = [
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_adverse_tone",
        "same_party_endorsement_proxy",
        "same_party_defense_proxy",
        "cross_party_attack_proxy",
        "cross_party_rebuttal_proxy",
    ]
    signed_columns = ["same_party_net_tone", "cross_party_net_tone", "party_tone_contrast", "party_stance_signal"]
    for column in nonnegative_columns + signed_columns:
        out[f"{column}_raw"] = out[column]
    for column in nonnegative_columns:
        max_by_election = out.groupby("election_id")[column].transform("max")
        out[column] = np.where(max_by_election.to_numpy(float) > 0.0, out[column] / max_by_election, 0.0)
    for column in signed_columns:
        max_abs_by_election = out.groupby("election_id")[column].transform(lambda s: s.abs().max())
        out[column] = np.where(max_abs_by_election.to_numpy(float) > 0.0, out[column] / max_abs_by_election, 0.0)
    for column in [
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_net_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_net_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "party_stance_signal",
    ]:
        out[f"{column}_centered"] = out[column] - out.groupby("election_id")[column].transform("mean")
    return out.sort_values(["election_id", "slot"])


def _empty_row(candidate: object) -> dict[str, object]:
    election_id = str(getattr(candidate, "election_id", ""))
    return {
        "election_id": election_id,
        "slot": getattr(candidate, "slot", ""),
        "candidate_name": getattr(candidate, "candidate_name", ""),
        "candidate_bloc": getattr(candidate, "candidate_bloc", ""),
        "same_party_positive_tone": 0.0,
        "same_party_negative_tone": 0.0,
        "same_party_net_tone": 0.0,
        "same_party_supportive_tone": 0.0,
        "cross_party_positive_tone": 0.0,
        "cross_party_negative_tone": 0.0,
        "cross_party_net_tone": 0.0,
        "cross_party_adverse_tone": 0.0,
        "party_tone_contrast": 0.0,
        "same_party_endorsement_proxy": 0.0,
        "same_party_defense_proxy": 0.0,
        "cross_party_attack_proxy": 0.0,
        "cross_party_rebuttal_proxy": 0.0,
        "party_stance_signal": 0.0,
        "manual_valence_coverage": 0.0,
        "available_date": cutoff_dates_as_strings(ELECTION_DATES).get(election_id, ""),
        "confidence": 0.0,
        "notes": "No candidate issue rows available",
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = build_tone_gap()
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved: {OUTPUT}")
    if not out.empty:
        print(
            out[
                [
                    "election_id",
                    "slot",
                    "candidate_name",
                    "same_party_supportive_tone",
                    "cross_party_adverse_tone",
                    "party_tone_contrast",
                    "confidence",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
