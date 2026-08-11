"""Strict nested comparison of alternative electorate mass definitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_electorate_layers as base_eval  # noqa: E402
import evaluate_nested_electorate_profiles as profile_eval  # noqa: E402


MASS_PROFILES = (
    "legacy",
    "direct_party_layers",
    "durable_floor",
    "broad_critical",
    "durable_floor_broad_critical",
)
PRIMARY_MASS_PROFILE = "direct_party_layers"
OUTPUT_DIR = ROOT / "outputs" / "electorate_mass_profile_experiment"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    comparisons: list[pd.DataFrame] = []
    configs: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    mass_summaries: list[pd.DataFrame] = []
    mass_shift_rows: list[dict[str, object]] = []
    legacy_masses: pd.DataFrame | None = None

    for mass_profile in MASS_PROFILES:
        frame = base_eval.prepare_frame(mass_profile=mass_profile)
        mass_keys = ["election_id", "region_id", "slot"]
        if mass_profile == "legacy":
            legacy_masses = frame[
                [*mass_keys, "core_voting_mass", "critical_voting_mass"]
            ].copy()
            mass_shift_rows.append(
                {
                    "mass_profile": mass_profile,
                    "maximum_core_reclassification_pp": 0.0,
                    "maximum_critical_reclassification_pp": 0.0,
                }
            )
        else:
            if legacy_masses is None:
                raise RuntimeError("legacy mass profile must be evaluated first")
            shifts = frame[
                [*mass_keys, "core_voting_mass", "critical_voting_mass"]
            ].merge(
                legacy_masses,
                on=mass_keys,
                suffixes=("", "_legacy"),
                validate="one_to_one",
            )
            mass_shift_rows.append(
                {
                    "mass_profile": mass_profile,
                    "maximum_core_reclassification_pp": float(
                        (shifts["core_voting_mass"] - shifts["core_voting_mass_legacy"])
                        .abs()
                        .max()
                        * 100.0
                    ),
                    "maximum_critical_reclassification_pp": float(
                        (
                            shifts["critical_voting_mass"]
                            - shifts["critical_voting_mass_legacy"]
                        )
                        .abs()
                        .max()
                        * 100.0
                    ),
                }
            )
        mass_columns = [
            "core_voting_mass",
            "critical_voting_mass",
            "swing_voting_mass",
            "direct_party_reliability",
        ]
        mass_summary = frame.groupby("election_id", as_index=False)[mass_columns].mean()
        mass_summary.insert(0, "mass_profile", mass_profile)
        mass_summaries.append(mass_summary)

        result, comparison, profile_configs, profile_traces = profile_eval._evaluate_profile(
            frame,
            "critical_defection",
            {},
        )
        result["mass_profile"] = mass_profile
        result["predeclared_primary_mass_profile"] = mass_profile == PRIMARY_MASS_PROFILE
        comparison.insert(0, "mass_profile", mass_profile)
        profile_configs.insert(0, "mass_profile", mass_profile)
        profile_traces.insert(0, "mass_profile", mass_profile)
        results.append(result)
        comparisons.append(comparison)
        configs.append(profile_configs)
        traces.append(profile_traces)

    all_comparisons = pd.concat(comparisons, ignore_index=True)
    reference = all_comparisons.loc[
        all_comparisons["mass_profile"].eq("direct_party_layers"),
        ["election_id", "existing_nested_mae_pp"],
    ].rename(columns={"existing_nested_mae_pp": "active_reference_mae_pp"})
    candidate_comparison = all_comparisons.merge(reference, on="election_id", how="left")
    candidate_comparison["improvement_vs_active_pp"] = (
        candidate_comparison["active_reference_mae_pp"]
        - candidate_comparison["existing_nested_mae_pp"]
    )

    profile_summary_rows = []
    for mass_profile, group in candidate_comparison.groupby("mass_profile", sort=False):
        improvement = float(group["improvement_vs_active_pp"].mean())
        max_worsening = float(
            (-group["improvement_vs_active_pp"]).clip(lower=0.0).max()
        )
        profile_summary_rows.append(
            {
                "mass_profile": mass_profile,
                "predeclared_primary": mass_profile == PRIMARY_MASS_PROFILE,
                "active_reference_nested_macro_mae_pp": float(
                    group["active_reference_mae_pp"].mean()
                ),
                "candidate_nested_macro_mae_pp": float(
                    group["existing_nested_mae_pp"].mean()
                ),
                "improvement_vs_active_pp": improvement,
                "maximum_outer_worsening_pp": max_worsening,
                "improves_at_least_three_elections": int(
                    (group["improvement_vs_active_pp"] > 1e-12).sum()
                ) >= 3,
                "passes_0_01pp_adoption_gate": (
                    improvement >= 0.01 and max_worsening <= 0.05
                ),
            }
        )
    profile_summary = pd.DataFrame(profile_summary_rows).merge(
        pd.DataFrame(mass_shift_rows), on="mass_profile", how="left"
    )
    profile_summary["passes_layer_reclassification_cap"] = (
        profile_summary[[
            "maximum_core_reclassification_pp",
            "maximum_critical_reclassification_pp",
        ]].max(axis=1)
        <= 3.0 + 1e-9
    )
    profile_summary["passes_0_01pp_adoption_gate"] &= profile_summary[
        "passes_layer_reclassification_cap"
    ]
    profile_summary_rows = profile_summary.to_dict(orient="records")
    primary = profile_summary.loc[
        profile_summary["mass_profile"].eq(PRIMARY_MASS_PROFILE)
    ].iloc[0]
    legacy_rows = all_comparisons.loc[all_comparisons["mass_profile"].eq("legacy")]
    direct_rows = all_comparisons.loc[
        all_comparisons["mass_profile"].eq("direct_party_layers")
    ]
    legacy_macro = float(legacy_rows["existing_nested_mae_pp"].mean())
    direct_macro = float(direct_rows["existing_nested_mae_pp"].mean())
    direct_by_election = direct_rows.set_index("election_id")["existing_nested_mae_pp"]
    legacy_by_election = legacy_rows.set_index("election_id")["existing_nested_mae_pp"]
    direct_improvement = legacy_by_election - direct_by_election
    structural_adoption_gates = {
        "strict_nested_noninferior_to_legacy": direct_macro <= legacy_macro + 1e-12,
        "maximum_election_worsening_at_most_0_005pp": float(
            (-direct_improvement).clip(lower=0.0).max()
        ) <= 0.005,
        "final_layer_reclassification_within_3pp": bool(
            primary["passes_layer_reclassification_cap"]
        ),
        "post_2022_presidential_outcomes_excluded": True,
    }
    payload = {
        "scope": {
            "scored_elections": list(base_eval.ALLOWED_ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "metric": "contest-vote weighted row MAE within election, equal-election macro",
            "mass_comparison_uses_response_separation": False,
        },
        "predeclared_primary_mass_profile": PRIMARY_MASS_PROFILE,
        "mass_profile_definitions": {
            "legacy": "25th-percentile core with direct-party mean LCB; persistent gap is critical",
            "direct_party_layers": "direct party ballots define party base with a 2.5pp raw/3pp final reclassification cap; candidate excess remains swing/personal vote",
            "durable_floor": "10th-percentile core without mean LCB; persistent gap is critical",
            "broad_critical": "legacy core; full recent-base-minus-core gap is critical",
            "durable_floor_broad_critical": "10th-percentile core; full recent-base-minus-core gap is critical",
        },
        "profile_results": profile_summary_rows,
        "adopt_primary_into_active_engine": bool(primary["passes_0_01pp_adoption_gate"]),
        "legacy_nested_macro_mae_pp": legacy_macro,
        "direct_party_layers_nested_macro_mae_pp": direct_macro,
        "direct_party_layers_improvement_pp": legacy_macro - direct_macro,
        "structural_adoption_gates": structural_adoption_gates,
        "adopt_primary_as_noninferior_structural_refinement": bool(
            all(structural_adoption_gates.values())
        ),
        "caveat": (
            "The mass profiles are theory-fixed ablations. Only the predeclared primary can be "
            "adopted; observed best-of-four selection is not permitted."
        ),
    }
    profile_summary.to_csv(
        OUTPUT_DIR / "profile_summary.csv", index=False, encoding="utf-8-sig"
    )
    candidate_comparison.to_csv(
        OUTPUT_DIR / "election_comparison.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(mass_summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / "mass_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(configs, ignore_index=True).to_csv(
        OUTPUT_DIR / "outer_selected_configs.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(traces, ignore_index=True).to_csv(
        OUTPUT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
