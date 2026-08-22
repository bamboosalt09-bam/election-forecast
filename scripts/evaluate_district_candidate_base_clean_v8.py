"""Clean strict-nested ablation of only the candidate regional-base input."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "district_candidate_base_clean_v8_ablation"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
DISTRICT_BASE = (
    ROOT
    / "outputs"
    / "district_reconstructed_candidate_base_v6"
    / "candidate_regional_base.csv"
)
VARIANTS = {
    "district_exact_0.40": {
        "response": None,
        "rejection_routing": False,
    },
    "district_exact_balanced_0.60": {
        "response": {
            "contest_regime_expansion_gain": 0.60,
            "contest_regime_log_shift_cap": 0.60,
            "contest_regime_swing_log_shift_cap": 0.75,
        },
        "rejection_routing": False,
    },
    "district_exact_balanced_0.60_routed": {
        "response": {
            "contest_regime_expansion_gain": 0.60,
            "contest_regime_log_shift_cap": 0.60,
            "contest_regime_swing_log_shift_cap": 0.75,
        },
        "rejection_routing": True,
    },
}


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _run_variant(
    label: str,
    response: dict[str, float] | None,
    *,
    rejection_routing: bool,
    candidate_base_path: Path = DISTRICT_BASE,
    chungcheong_alignment_path: Path | None = None,
    third_profile_path: Path | None = None,
    third_pressure_path: Path | None = None,
    config_path: Path | None = None,
    run_dir_override: Path | None = None,
    assignment_dir_override: Path | None = None,
    regenerate_issue_seeds_enabled: bool = False,
    output_root: Path = OUTPUT_DIR,
) -> Path:
    variant_root = output_root / label
    assignment_dir = (
        variant_root / f"slot_assignment_{label}"
        if assignment_dir_override is None
        else Path(assignment_dir_override)
    )
    run_dir = (
        variant_root / "active_run"
        if run_dir_override is None
        else Path(run_dir_override)
    )
    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (active, "ASSIGNMENT_DIR", assignment_dir),
        (
            active.nested,
            "ASSIGNMENT_PATH",
            assignment_dir / "candidate_slot_assignments_v2.csv",
        ),
        (active, "CANDIDATE_REGIONAL_BASE", candidate_base_path),
    ]
    if not regenerate_issue_seeds_enabled:
        attributes.append((active, "regenerate_issue_seeds", lambda: None))
    if chungcheong_alignment_path is not None:
        attributes.append(
            (active, "CHUNGCHEONG_ALIGNMENT", chungcheong_alignment_path)
        )
    for engine in engines:
        attributes.append(
            (engine, "CANDIDATE_REGIONAL_BASE", str(candidate_base_path))
        )
        if third_profile_path is not None:
            attributes.append(
                (engine, "THIRD_CANDIDATE_PROFILE", str(third_profile_path))
            )
        if third_pressure_path is not None:
            attributes.append(
                (engine, "THIRD_CANDIDATE_PRESSURE", str(third_pressure_path))
            )
    original_load_policy = active.load_policy
    if config_path is not None:
        effective_config_path = Path(config_path)

        def load_selected_policy(path=effective_config_path):
            return original_load_policy(effective_config_path)

        attributes.extend(
            [
                (active, "CONFIG_PATH", effective_config_path),
                (active, "load_policy", load_selected_policy),
            ]
        )
    if response is not None:

        def load_policy_with_response(
            path=active.CONFIG_PATH,
            *,
            _response=response,
        ):
            policy = deepcopy(original_load_policy(path))
            policy["postprocess"].update(_response)
            return policy

        attributes.append((active, "load_policy", load_policy_with_response))

    with patching.patched(attributes):
        active.run(
            output_dir=run_dir,
            rejection_beneficiary_routing_enabled=rejection_routing,
        )
    return run_dir


def _manifest_diff(path: Path) -> pd.DataFrame:
    active_manifest = pd.read_csv(
        ACTIVE_DIR / "input_manifest.csv", encoding="utf-8-sig"
    )
    variant_manifest = pd.read_csv(path / "input_manifest.csv", encoding="utf-8-sig")
    active_manifest["basename"] = active_manifest["path"].map(
        lambda value: Path(str(value)).name
    )
    variant_manifest["basename"] = variant_manifest["path"].map(
        lambda value: Path(str(value)).name
    )
    active_lookup = active_manifest.drop_duplicates("basename").set_index("basename")
    variant_lookup = variant_manifest.drop_duplicates("basename").set_index("basename")
    rows: list[dict[str, object]] = []
    for basename in sorted(set(active_lookup.index) | set(variant_lookup.index)):
        active_hash = (
            str(active_lookup.at[basename, "sha256"])
            if basename in active_lookup.index
            else ""
        )
        variant_hash = (
            str(variant_lookup.at[basename, "sha256"])
            if basename in variant_lookup.index
            else ""
        )
        if active_hash != variant_hash:
            rows.append(
                {
                    "basename": basename,
                    "active_sha256": active_hash,
                    "variant_sha256": variant_hash,
                    "expected_difference": basename
                    in {
                        "candidate_regional_base.csv",
                        "candidate_slot_assignments_v2.csv",
                        "active_presidential_model_v16.json",
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {
        label: _run_variant(
            label,
            config["response"],
            rejection_routing=bool(config["rejection_routing"]),
        )
        for label, config in VARIANTS.items()
    }

    summary_rows = [{"variant_label": "active_v16", **_metrics(ACTIVE_DIR)}]
    election_frames: list[pd.DataFrame] = []
    national_frames: list[pd.DataFrame] = []
    for label, path in [("active_v16", ACTIVE_DIR), *runs.items()]:
        if label != "active_v16":
            summary_rows.append({"variant_label": label, **_metrics(path)})
        by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        by_election["variant_label"] = label
        election_frames.append(by_election)
        national = pd.read_csv(
            path / "national_predictions.csv", encoding="utf-8-sig"
        )
        national["variant_label"] = label
        national_frames.append(national)

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    national.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    manifest_diff = _manifest_diff(runs["district_exact_0.40"])
    manifest_diff.to_csv(
        OUTPUT_DIR / "input_manifest_diff.csv", index=False, encoding="utf-8-sig"
    )
    unexpected = (
        manifest_diff.loc[~manifest_diff["expected_difference"]]
        if not manifest_diff.empty
        else manifest_diff
    )
    decision = {
        "experiment": "district_candidate_base_clean_v8",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "active_model_changed": False,
        "role_aware_assignment": False,
        "candidate_context_override": False,
        "clean_0_40_unexpected_input_differences": int(len(unexpected)),
        "balanced_0_60_parameter_selection_is_outcome_aware": True,
        "promotion_decision": "hold_as_candidate",
        "promotion_reason": (
            "Aggregate and 2007/2017 errors improve, but the 0.60 response was "
            "selected on the same through-2022 elections and constituency "
            "evidence still needs province-footprint control."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(
        by_election.loc[
            by_election["election_id"].eq("pres_2017"),
            [
                "variant_label",
                "regional_weighted_mae_pp",
                "national_candidate_mae_pp",
            ],
        ].to_string(index=False)
    )
    print()
    print(manifest_diff.to_string(index=False))


if __name__ == "__main__":
    main()
