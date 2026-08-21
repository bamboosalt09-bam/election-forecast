"""Outcome-free compilers for the remaining presidential control inputs.

The functions in this module never read target-election vote outcomes.  They
turn dated factual registries, Assembly evidence, and strictly prior election
artifacts into the CSV schemas already consumed by the forecast engine.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from presidential_issue_engine.incumbent_shock_adjustment import (
    compile_government_burden_scores,
)
from presidential_issue_engine.point_in_time import filter_available_by_election
from presidential_issue_engine.speech_derived_third_pressure import (
    build_automatic_third_candidate_pressure,
)


SCHEMA_VERSION = "automatic_controls_v22"
SHOCK_CLASS_LEVEL = {
    "institutional_crisis": 1.00,
    "accountability_scandal": 0.60,
    "incumbent_assessment": 0.60,
    "political_realignment": 0.50,
    "diffuse_issue_environment": 0.35,
}
# The two evidence thresholds an election must clear to be classified as an
# institutional crisis. Named because anything reasoning about how far an
# election sits above or below the crisis boundary has to use the same numbers
# the classifier uses, rather than introducing a threshold of its own.
CRISIS_MIN_REGIME_EVIDENCE = 0.65
CRISIS_ACCOUNTABILITY = 0.75
SHOCK_CLASS_INTENSITY = {
    "institutional_crisis": 2.00,
    "accountability_scandal": 1.00,
    "incumbent_assessment": 1.00,
    "political_realignment": 0.75,
    "diffuse_issue_environment": 0.50,
}
PROFILE_FIELDS = [
    "viability",
    "centrist_appeal",
    "anti_major_party_appeal",
    "regional_base_overlap",
]
DIRECT_PARTY_TYPES = {
    "national_assembly_pr",
    "assembly_pr",
    "metro_council_pr",
    "local_council_pr",
}


def _bounded(values: pd.Series | float, lower: float = 0.0, upper: float = 1.0):
    return np.clip(pd.to_numeric(values, errors="coerce"), lower, upper)


def build_automatic_mega_taxonomy(
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive shock type and numeric character from Assembly diagnostics.

    The classification thresholds are universal evidence thresholds.  They do
    not contain election ids or observed vote errors.
    """

    required = {
        "election_id",
        "source_rows",
        "salience_component",
        "severity_component",
        "breadth_component",
        "accountability_component",
        "joint_evidence",
        "available_date",
    }
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = diagnostics.copy()
    for column in required - {"election_id", "available_date"}:
        work[column] = _bounded(work[column]) if column != "source_rows" else pd.to_numeric(
            work[column], errors="coerce"
        ).fillna(0.0)

    minimum_regime_evidence = work[
        ["salience_component", "severity_component", "breadth_component"]
    ].min(axis=1)
    shock_type = np.select(
        [
            minimum_regime_evidence.ge(CRISIS_MIN_REGIME_EVIDENCE)
            & work["accountability_component"].ge(CRISIS_ACCOUNTABILITY),
            work["accountability_component"].ge(0.80)
            & work["salience_component"].ge(0.40),
            work["salience_component"].ge(0.60)
            & work["breadth_component"].ge(0.60),
            work["salience_component"].ge(0.25)
            & work["breadth_component"].ge(0.50),
        ],
        [
            "institutional_crisis",
            "accountability_scandal",
            "incumbent_assessment",
            "political_realignment",
        ],
        default="diffuse_issue_environment",
    )
    class_level = pd.Series(shock_type, index=work.index).map(SHOCK_CLASS_LEVEL)
    # Once the evidence crosses a semantic-class threshold, use the same
    # class-level shock scale in every election.  Raw mention volume remains
    # in the numeric taxonomy and confidence audit, but no longer shrinks an
    # institutional crisis back toward an ordinary campaign issue.
    class_intensity = pd.Series(shock_type, index=work.index).map(
        SHOCK_CLASS_INTENSITY
    )
    evidence_volume = np.clip(np.log1p(work["source_rows"]) / np.log1p(50_000), 0.0, 1.0)
    confidence = np.sqrt(evidence_volume * work["breadth_component"].clip(0.0, 1.0))
    severity = np.sqrt(
        work["joint_evidence"].clip(0.0, 1.0)
        * work["accountability_component"].clip(0.0, 1.0)
    )
    taxonomy = pd.DataFrame(
        {
            "election_id": work["election_id"].astype(str),
            "mega_event": "assembly_regime_accountability_environment",
            "shock_type": shock_type,
            "severity": severity,
            "national_scope": work["breadth_component"],
            "persistence": np.sqrt(
                work["salience_component"] * work["breadth_component"]
            ),
            "polarization": (
                work["salience_component"] + work["accountability_component"]
            )
            / 2.0,
            "target_specificity": work["accountability_component"],
            "available_date": work["available_date"].astype(str),
            "confidence": confidence,
            "notes": (
                "Assembly-derived universal shock taxonomy; no election-specific "
                "numeric seed or vote outcome"
            ),
        }
    )
    intensity = pd.DataFrame(
        {
            "election_id": work["election_id"].astype(str),
            "mega_issue_intensity": class_intensity,
            "available_date": work["available_date"].astype(str),
            "notes": (
                "Assembly evidence gated by an automatically classified universal "
                "shock type"
            ),
        }
    )
    audit = work.copy()
    audit["automatic_shock_type"] = shock_type
    audit["automatic_event_class_level"] = class_level
    audit["automatic_event_class_intensity"] = class_intensity
    audit["automatic_taxonomy_confidence"] = confidence
    audit["automatic_mega_issue_intensity"] = intensity["mega_issue_intensity"]
    audit["target_outcome_used"] = False
    return (
        taxonomy.sort_values("election_id").reset_index(drop=True),
        intensity.sort_values("election_id").reset_index(drop=True),
        audit.sort_values("election_id").reset_index(drop=True),
    )


def build_automatic_responsibility_alignments(
    candidate_issue_profile: pd.DataFrame,
    candidate_context: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine structural incumbency with discourse evidence by domain."""

    burden = compile_government_burden_scores(candidate_issue_profile, dict(election_dates))
    if burden.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    context = filter_available_by_election(
        candidate_context.copy(),
        dict(election_dates),
        source_name="automatic_responsibility_candidate_context",
    )
    context = (
        context.sort_values("available_date")
        .drop_duplicates(["election_id", "slot"], keep="last")
        .reset_index(drop=True)
    )
    context["organization_strength"] = _bounded(
        context.get("organization_strength", pd.Series(0.0, index=context.index))
    )

    profile = filter_available_by_election(
        candidate_issue_profile.copy(),
        dict(election_dates),
        source_name="automatic_responsibility_discourse",
    )
    profile["target_absolute_evidence"] = pd.to_numeric(
        profile.get("target_absolute_evidence"), errors="coerce"
    ).fillna(0.0)
    profile["target_attribution_confidence"] = _bounded(
        profile.get("target_attribution_confidence", pd.Series(0.0, index=profile.index))
    )
    government = profile.loc[
        profile.get("target_source_types", pd.Series("", index=profile.index))
        .fillna("")
        .astype(str)
        .str.contains(r"(?:^|\|)government(?:\||$)", regex=True)
    ].copy()

    domain_issues = {
        "economic": {
            "economy_growth",
            "inflation_livelihood",
            "jobs_labor",
            "external_shock",
        },
        "housing": {"housing"},
    }
    incumbent = burden.sort_values(
        ["election_id", "government_evidence_weight"], ascending=[True, False]
    ).drop_duplicates("election_id")
    incumbent_slots = incumbent.set_index("election_id")["slot"].astype(str).to_dict()
    burden_dates = (
        government.groupby("election_id")["available_date"].max().astype(str).to_dict()
        if not government.empty
        else {}
    )

    outputs: dict[str, list[dict[str, object]]] = {"economic": [], "housing": []}
    audits: list[dict[str, object]] = []
    for election_id, election_rows in context.groupby("election_id", sort=True):
        incumbent_slot = incumbent_slots.get(str(election_id))
        if incumbent_slot is None:
            continue
        max_organization = max(float(election_rows["organization_strength"].max()), 1e-9)
        for domain, issues in domain_issues.items():
            evidence = government.loc[
                government["election_id"].astype(str).eq(str(election_id))
                & government["slot"].astype(str).eq(incumbent_slot)
                & government["issue_name"].astype(str).isin(issues)
            ]
            discourse_mass = float(
                (
                    evidence["target_absolute_evidence"]
                    * evidence["target_attribution_confidence"]
                ).sum()
            )
            discourse_strength = float(np.tanh(np.log1p(max(discourse_mass, 0.0))))
            responsibility_scale = 0.75 + 0.25 * discourse_strength
            for row in election_rows.itertuples(index=False):
                slot = str(row.slot)
                organization_ratio = float(row.organization_strength) / max_organization
                score = (
                    responsibility_scale
                    if slot == incumbent_slot
                    else -responsibility_scale * organization_ratio
                )
                available_date = pd.Timestamp(
                    max(
                        pd.Timestamp(row.available_date),
                        pd.Timestamp(
                            burden_dates.get(
                                str(election_id), str(row.available_date)
                            )
                        ),
                    )
                ).date().isoformat()
                outputs[domain].append(
                    {
                        "election_id": str(election_id),
                        "slot": slot,
                        f"{domain}_responsibility_score": float(np.clip(score, -1.0, 1.0)),
                        "available_date": available_date,
                        "source_note": (
                            "Automatic structural incumbent responsibility plus "
                            "explicit government-target discourse evidence"
                        ),
                    }
                )
                audits.append(
                    {
                        "election_id": str(election_id),
                        "slot": slot,
                        "domain": domain,
                        "incumbent_slot": incumbent_slot,
                        "organization_ratio": organization_ratio,
                        "discourse_mass": discourse_mass,
                        "discourse_strength": discourse_strength,
                        "responsibility_score": score,
                        "target_outcome_used": False,
                    }
                )
    economic = pd.DataFrame(outputs["economic"])
    housing = pd.DataFrame(outputs["housing"])
    return economic, housing, pd.DataFrame(audits)


def build_automatic_generation_weights(
    official_history: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only the latest turnout composition published before each target."""

    history = official_history.copy()
    for column in ["event_date", "published_date"]:
        history[column] = pd.to_datetime(history[column], errors="coerce")
    for column in ["young_weight", "middle_weight", "senior_weight"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(
        subset=[
            "event_date",
            "published_date",
            "young_weight",
            "middle_weight",
            "senior_weight",
        ]
    )
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for election_id, cutoff_value in election_dates.items():
        cutoff = pd.Timestamp(cutoff_value)
        eligible = history.loc[
            history["event_date"].lt(cutoff)
            & history["published_date"].lt(cutoff)
        ].sort_values(["event_date", "published_date"])
        if eligible.empty:
            weights = np.asarray([0.25, 0.50, 0.25], dtype=float)
            available_date = "1990-01-01"
            source_election = "universal_uninformed_prior"
            confidence = 0.20
        else:
            selected = eligible.iloc[-1]
            weights = selected[
                ["young_weight", "middle_weight", "senior_weight"]
            ].to_numpy(float)
            available_date = pd.Timestamp(selected["published_date"]).date().isoformat()
            source_election = str(selected["source_election_id"])
            confidence = float(np.exp(-max((cutoff - selected["event_date"]).days, 0) / (365.25 * 12.0)))
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()
        rows.append(
            {
                "election_id": str(election_id),
                "young_weight": float(weights[0]),
                "middle_weight": float(weights[1]),
                "senior_weight": float(weights[2]),
                "available_date": available_date,
                "notes": (
                    f"Latest official age-turnout composition strictly before target: {source_election}"
                ),
            }
        )
        audit.append(
            {
                **rows[-1],
                "source_election_id": source_election,
                "lag_confidence": confidence,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit)


def build_full_automatic_third_profile(
    active_profile: pd.DataFrame,
    election_derived_profile: pd.DataFrame,
    preliminary_profile: pd.DataFrame,
    preliminary_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace manual profile fields where dated structural evidence exists."""

    output = active_profile.copy()
    audit: list[dict[str, object]] = []
    derived_lookup = election_derived_profile.set_index(["election_id", "slot"])
    preliminary_lookup = preliminary_profile.set_index(["election_id", "slot"])
    assignment_lookup = preliminary_assignments.loc[
        preliminary_assignments["source_slot"].astype(str).eq("C")
    ].set_index(["election_id", "source_slot"])

    for index, row in output.iterrows():
        key = (str(row["election_id"]), str(row["slot"]))
        source = "low_information_structural_fallback"
        replacement: dict[str, object] = {}
        if key in derived_lookup.index:
            selected = derived_lookup.loc[key]
            replacement = {field: float(selected[field]) for field in PROFILE_FIELDS}
            replacement["candidate_name"] = str(selected["candidate_name"])
            replacement["confidence"] = float(selected["confidence"])
            replacement["available_date"] = str(selected["available_date"])
            source = "election_and_assembly_derived"
        elif key in preliminary_lookup.index:
            selected = preliminary_lookup.loc[key]
            replacement = {
                "viability": float(selected["viability"]),
                "regional_base_overlap": float(selected["regional_base_overlap"]),
                "candidate_name": str(selected["candidate_name"]),
                "confidence": float(selected["confidence"]),
                "available_date": str(selected["available_date"]),
            }
            source = "strictly_prior_candidate_profile"
        elif key in assignment_lookup.index:
            selected = assignment_lookup.loc[key]
            pre_share = float(selected["pre_withdrawal_mean_share"])
            replacement = {
                "candidate_name": str(selected["candidate_name"]),
                "viability": float(np.clip(pre_share / 0.25, 0.0, 1.0)),
                "centrist_appeal": 0.50,
                "anti_major_party_appeal": 0.50,
                "regional_base_overlap": 0.0,
                "confidence": 0.25,
                "available_date": str(selected["available_date"]),
            }
        for column, value in replacement.items():
            output.at[index, column] = value
        output.at[index, "notes"] = (
            f"Automatic V22 profile source={source}; no target-election vote outcome"
        )
        audit.append(
            {
                "election_id": key[0],
                "slot": key[1],
                "candidate_name": output.at[index, "candidate_name"],
                "source": source,
                "fields_replaced": "|".join(sorted(replacement)),
                "target_outcome_used": False,
            }
        )
    return output, pd.DataFrame(audit)


def build_automatic_withdrawn_landscape(
    base_landscape: pd.DataFrame,
    third_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace manual withdrawn-person axes with profile-derived low-rank axes."""

    out = base_landscape.copy()
    profiles = third_profile.set_index(["election_id", "slot"])
    audits: list[dict[str, object]] = []
    role = out.get("candidate_role", pd.Series("", index=out.index)).astype(str)
    for index in out.index[role.str.contains("withdraw", case=False, na=False)]:
        key = (str(out.at[index, "election_id"]), str(out.at[index, "slot"]))
        if key not in profiles.index:
            continue
        profile = profiles.loc[key]
        centrist = float(profile["centrist_appeal"])
        anti_major = float(profile["anti_major_party_appeal"])
        regional = float(profile["regional_base_overlap"])
        ideology_mass = max(1.0 - centrist, 0.0)
        values = {
            "conservative": ideology_mass / 2.0,
            "liberal": ideology_mass / 2.0,
            "progressive": ideology_mass / 4.0,
            "centrist": centrist,
            "anti_establishment": anti_major,
            "reform": (centrist + anti_major) / 2.0,
            "regionalist": regional,
            "confidence": min(float(profile["confidence"]), 0.50),
            "available_date": str(profile["available_date"]),
            "candidate_name": str(profile["candidate_name"]),
            "notes": "Automatic low-rank withdrawn-candidate landscape from the third profile",
        }
        for column, value in values.items():
            out.at[index, column] = value
        audits.append(
            {
                "election_id": key[0],
                "slot": key[1],
                "candidate_name": values["candidate_name"],
                "target_outcome_used": False,
            }
        )
    return out, pd.DataFrame(audits)


def build_third_pressure_v22(
    third_profile: pd.DataFrame,
    speech_context: pd.DataFrame,
    political_landscape: pd.DataFrame,
    candidate_issue_profile: pd.DataFrame,
    mega_intensity: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a generic high-shock incumbent-lane term to automatic pressure."""

    pressure = build_automatic_third_candidate_pressure(
        third_profile,
        speech_context,
        political_landscape,
        election_dates,
    )
    if pressure.empty:
        return pressure, pd.DataFrame()
    burden = compile_government_burden_scores(candidate_issue_profile, dict(election_dates))
    burden = burden.rename(columns={"slot": "source_slot"})
    intensity = mega_intensity[["election_id", "mega_issue_intensity"]].copy()
    out = pressure.merge(
        burden[
            ["election_id", "source_slot", "government_rejection_strength"]
        ],
        on=["election_id", "source_slot"],
        how="left",
    ).merge(intensity, on="election_id", how="left")
    out["incumbent_responsibility_lane"] = out[
        "government_rejection_strength"
    ].notna().astype(float)
    out["government_rejection_strength"] = pd.to_numeric(
        out["government_rejection_strength"], errors="coerce"
    ).fillna(0.0)
    out["mega_issue_intensity"] = pd.to_numeric(
        out["mega_issue_intensity"], errors="coerce"
    ).fillna(1.0)
    out["regime_lane_multiplier"] = 1.0 + (
        out["mega_issue_intensity"] - 1.0
    ).clip(lower=0.0) * (1.0 + out["government_rejection_strength"]) * out[
        "incumbent_responsibility_lane"
    ]
    out["lane_affinity_v22"] = out["lane_affinity"] * out["regime_lane_multiplier"]
    lane_total = out.groupby(["election_id", "slot"])["lane_affinity_v22"].transform("sum")
    out["lane_share"] = out["lane_affinity_v22"] / lane_total.replace(0.0, np.nan)
    out["lane_share"] = out["lane_share"].fillna(0.0)
    out["issue_environment_conversion_scale"] = out[
        "mega_issue_intensity"
    ].clip(0.50, 1.25)
    out["draw_propensity_v22"] = (
        out["draw_propensity"] * out["issue_environment_conversion_scale"]
    ).clip(0.0, 1.0)
    out["transfer_pressure"] = (
        out["draw_propensity_v22"] * out["lane_share"]
    ).clip(0.0, 1.0)
    out["notes"] = (
        "Automatic political-axis pressure with high-shock incumbent-lane vulnerability"
    )
    out["derivation_version"] = SCHEMA_VERSION + "_third_pressure"
    audit = out.copy()
    audit["target_outcome_used"] = False
    return out[pressure.columns].copy(), audit


def build_behavioral_party_transitions(
    transitions: pd.DataFrame,
    exact_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create dated retention edges from strictly prior direct-party ballots."""

    events = exact_events.loc[
        exact_events["election_type"].astype(str).isin(DIRECT_PARTY_TYPES)
    ].copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["regional_share"] = pd.to_numeric(
        events["regional_share"], errors="coerce"
    ).fillna(0.0)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for transition in transitions.itertuples(index=False):
        predecessor = str(transition.predecessor_party).strip()
        successor = str(transition.successor_party).strip()
        effective = pd.Timestamp(transition.effective_date)
        relation = str(transition.relation_type)
        structural_prior = 0.85 if relation == "rename" else 0.60
        base = transition._asdict()
        base["continuity"] = structural_prior
        base["confidence"] = 0.50
        base["notes"] = (
            str(base.get("notes", ""))
            + "; universal legal-transition retention prior"
        )
        rows.append(base)

        party_names = events["source_party_names"].fillna("").astype(str)
        predecessor_mask = party_names.str.split("|").map(
            lambda values: predecessor in {value.strip() for value in values}
        )
        successor_mask = party_names.str.split("|").map(
            lambda values: successor in {value.strip() for value in values}
        )
        before = events.loc[
            predecessor_mask & events["event_date"].lt(effective)
        ].copy()
        after = events.loc[
            successor_mask & events["event_date"].ge(effective)
        ].copy()
        if before.empty or after.empty:
            audits.append(
                {
                    "predecessor_party": predecessor,
                    "successor_party": successor,
                    "status": "structural_prior_only",
                    "target_outcome_used": False,
                }
            )
            continue
        before_date = before["event_date"].max()
        after_date = after["event_date"].min()
        before = before.loc[before["event_date"].eq(before_date)]
        after = after.loc[after["event_date"].eq(after_date)]
        left = before.groupby("region_id")["regional_share"].sum()
        right = after.groupby("region_id")["regional_share"].sum()
        common = left.index.intersection(right.index)
        if len(common) < 3:
            continue
        left_values = left.loc[common].to_numpy(float)
        right_values = right.loc[common].to_numpy(float)
        left_centered = left_values - np.median(left_values)
        right_centered = right_values - np.median(right_values)
        denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        similarity = float(
            np.clip(
                np.dot(left_centered, right_centered) / denominator
                if denominator > 1e-12
                else 0.0,
                0.0,
                1.0,
            )
        )
        mass_retention = float(
            np.clip(right_values.mean() / max(left_values.mean(), 1e-9), 0.0, 1.0)
        )
        retention = float(np.sqrt(similarity * mass_retention))
        update = base.copy()
        update["effective_date"] = pd.Timestamp(after_date).date().isoformat()
        update["continuity"] = retention
        update["confidence"] = float(len(common) / (len(common) + 4.0))
        update["notes"] = (
            "Behavioral retention update from the first strictly later direct-party ballot"
        )
        rows.append(update)
        audits.append(
            {
                "predecessor_party": predecessor,
                "successor_party": successor,
                "status": "behavioral_update",
                "before_event_date": pd.Timestamp(before_date).date().isoformat(),
                "after_event_date": pd.Timestamp(after_date).date().isoformat(),
                "spatial_similarity": similarity,
                "mass_retention": mass_retention,
                "retention": retention,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audits)
