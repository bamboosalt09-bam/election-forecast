"""Build candidate party-speech context features from assembly posture.

This script does not reprocess raw assembly transcripts.  It uses the already
extracted 15th-22nd speaker-level issue matches and speaker/member-history
weights to estimate elite-discourse support around each candidate.
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

from assembly_bloc_issue_posture import (  # noqa: E402
    diagnostics_for,
    load_assembly_bloc_issue_posture,
    vector_for,
)
from news_collector.sources.member_party import party_bloc  # noqa: E402
from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402


RESULTS = ROOT / "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
CANDIDATE_ISSUES = ROOT / "data/candidate_issue_link.csv"
CANDIDATE_PROFILE = ROOT / "data/raw/candidate_issue_profile.csv"
OUTPUT = ROOT / "data/raw/candidate_party_speech_context.csv"

ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}

RISK_ISSUES = {"corruption_integrity", "family_legal_risk", "gaffe_event"}
MAJOR_BLOCS = {"국민의힘", "더불어민주당", "보수정당계", "민주당계"}


def _cosine(left: pd.Series, right: pd.Series) -> float:
    left_values = left.fillna(0.0).to_numpy(float)
    right_values = right.fillna(0.0).to_numpy(float)
    denom = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(left_values, right_values) / denom, 0.0, 1.0))


def _candidate_issue_vectors() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if CANDIDATE_ISSUES.exists():
        frame = filter_available_by_election(
            pd.read_csv(CANDIDATE_ISSUES),
            ELECTION_DATES,
            source_name="candidate_issue_link",
        )
        frame["emphasis_within"] = pd.to_numeric(frame["emphasis_within"], errors="coerce").fillna(0.0)
        pieces.append(frame[["election_id", "slot", "issue_name", "emphasis_within"]].copy())
    if CANDIDATE_PROFILE.exists():
        profile = pd.read_csv(CANDIDATE_PROFILE)
        required = {"election_id", "slot", "issue_name", "association_strength", "confidence"}
        if required.issubset(profile.columns):
            profile = profile[["election_id", "slot", "issue_name", "association_strength", "confidence"]].copy()
            profile["emphasis_within"] = (
                pd.to_numeric(profile["association_strength"], errors="coerce").fillna(0.0).abs()
                * pd.to_numeric(profile["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            )
            pieces.append(profile[["election_id", "slot", "issue_name", "emphasis_within"]])
    if not pieces:
        return pd.DataFrame()
    frame = pd.concat(pieces, ignore_index=True)
    return frame.pivot_table(
        index=["election_id", "slot"],
        columns="issue_name",
        values="emphasis_within",
        aggfunc="sum",
        fill_value=0.0,
    )


def _formal_candidates() -> pd.DataFrame:
    results = pd.read_csv(RESULTS)
    results = results.loc[
        (results["slot"].astype(str) != "alpha")
        & results["is_active_slot"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    ].copy()
    candidates = (
        results[["election_id", "slot", "candidate_name", "party_name"]]
        .drop_duplicates(["election_id", "slot"])
        .copy()
    )
    candidates["bloc"] = candidates["party_name"].map(party_bloc).map(normalize_bloc)
    return candidates


def _organization_strength(party_name: str, bloc: str) -> float:
    if "무소속" in party_name or bloc == "무소속":
        return 0.15
    if bloc in MAJOR_BLOCS:
        return 0.95
    if bloc == "제3지대":
        return 0.60
    if bloc == "진보정당계":
        return 0.55
    return 0.50


def build_context() -> pd.DataFrame:
    candidate_vectors = _candidate_issue_vectors()
    bloc_issues, bloc_diag = load_assembly_bloc_issue_posture()
    all_issues = sorted(
        set(candidate_vectors.columns)
        | set(bloc_issues.get("issue_name", pd.Series(dtype=str)).dropna().astype(str))
    )
    candidate_vectors = candidate_vectors.reindex(columns=all_issues, fill_value=0.0)

    rows: list[dict[str, object]] = []
    for candidate in _formal_candidates().itertuples(index=False):
        election_id = str(candidate.election_id)
        slot = str(candidate.slot)
        key = (election_id, slot)
        candidate_vector = (
            candidate_vectors.loc[key] if key in candidate_vectors.index else pd.Series(0.0, index=all_issues)
        )
        bloc = str(candidate.bloc)
        bloc_vector = vector_for(bloc_issues, election_id, bloc).reindex(all_issues, fill_value=0.0)
        alignment = _cosine(candidate_vector, bloc_vector)
        diag = diagnostics_for(bloc_diag, election_id, bloc)
        convergence = float(diag.get("same_bloc_frame_convergence", 0.0)) if not diag.empty else 0.0
        coverage = float(diag.get("speaker_coverage", 0.0)) if not diag.empty else 0.0
        mapping_confidence = float(diag.get("avg_mapping_confidence", 0.0)) if not diag.empty else 0.0
        organization_strength = _organization_strength(str(candidate.party_name), bloc)
        outsider_status = 1.0 - organization_strength
        risk_attention = float(candidate_vector.reindex(RISK_ISSUES, fill_value=0.0).sum())
        cross_bloc_attack_pressure = risk_attention * (1.0 - 0.50 * organization_strength)
        intra_bloc_conflict_score = (1.0 - alignment) * organization_strength * 0.35 + outsider_status * 0.25
        party_elite_support_score = (
            alignment
            * (0.45 + 0.35 * organization_strength + 0.20 * convergence)
            * (0.70 + 0.20 * coverage + 0.10 * mapping_confidence)
        )
        party_elite_fragmentation_score = (
            0.55 * intra_bloc_conflict_score + 0.45 * cross_bloc_attack_pressure
        )
        party_context_support = party_elite_support_score - party_elite_fragmentation_score
        confidence = float(
            np.clip(
                0.25
                + 0.30 * coverage
                + 0.20 * alignment
                + 0.15 * mapping_confidence
                + 0.10 * organization_strength,
                0.0,
                0.85,
            )
        )
        rows.append(
            {
                "election_id": election_id,
                "slot": slot,
                "candidate_name": candidate.candidate_name,
                "bloc": bloc,
                "same_bloc_issue_alignment": alignment,
                "same_bloc_frame_convergence": convergence,
                "cross_bloc_attack_pressure": cross_bloc_attack_pressure,
                "intra_bloc_conflict_score": intra_bloc_conflict_score,
                "party_elite_support_score": party_elite_support_score,
                "party_elite_fragmentation_score": party_elite_fragmentation_score,
                "party_context_support": party_context_support,
                "organization_strength": organization_strength,
                "outsider_status": outsider_status,
                "available_date": cutoff_dates_as_strings(ELECTION_DATES).get(election_id, ""),
                "confidence": confidence,
                "notes": "Derived from candidate_issue_link plus 15th-22nd Assembly speaker/member-history issue posture; no transcript reprocessing",
            }
        )
    return pd.DataFrame(rows).sort_values(["election_id", "slot"])


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = build_context()
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved: {OUTPUT}")
    print(out[["election_id", "slot", "candidate_name", "party_context_support", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
