"""Strict nested ablation for the V23 unified candidate controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import unified_lineage_identity  # noqa: E402
from scripts import build_automatic_controls_v23 as builder  # noqa: E402
from scripts import evaluate_automatic_controls_v22 as eval22  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402
from scripts import run_active_presidential_model_v22 as active_v22  # noqa: E402


OUTPUT_DIR = Path(
    os.environ.get(
        "V23_OUTPUT_DIR",
        ROOT / "outputs" / "automatic_controls_v23_ablation",
    )
)
V22_DIR = ROOT / "outputs" / "automatic_controls_v22"
V23_DIR = ROOT / "outputs" / "automatic_controls_v23"
BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v22"
REGISTRY = V23_DIR / "withdrawal_transfer_registry.csv"

COMMON_SWITCHES = {
    "policy": True,
    "mega": True,
    "responsibility": True,
    "third": True,
}

VARIANTS = {
    "v22_reproduction": {
        "automatic_dir": V22_DIR,
        "registry": "",
        "generation": False,
    },
    "v23_profile_legacy_transfer": {
        "automatic_dir": V23_DIR,
        "registry": "",
        "generation": False,
    },
    "v23_profile_legacy_transfer_generation": {
        "automatic_dir": V23_DIR,
        "registry": "",
        "generation": True,
    },
    "v23_registry_v22_profile": {
        "automatic_dir": V22_DIR,
        "registry": str(REGISTRY),
        "generation": False,
    },
    "v23_unified_profile_transfer": {
        "automatic_dir": V23_DIR,
        "registry": str(REGISTRY),
        "generation": False,
    },
    "v23_unified_profile_transfer_generation": {
        "automatic_dir": V23_DIR,
        "registry": str(REGISTRY),
        "generation": True,
    },
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _run_variant(
    label: str,
    spec: dict[str, object],
    exact_events: pd.DataFrame,
) -> Path:
    switches = dict(COMMON_SWITCHES)
    switches["generation"] = bool(spec["generation"])
    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (eval22, "AUTOMATIC_DIR", Path(spec["automatic_dir"])),
        (eval22, "OUTPUT_DIR", OUTPUT_DIR),
        (eval22, "CONFIG", active_v22.CONFIG),
    ]
    attributes.extend(
        (engine, "WITHDRAWAL_TRANSFER_REGISTRY", str(spec["registry"]))
        for engine in engines
    )
    with patching.patched(attributes):
        return eval22._run_variant(label, switches, exact_events)


def main() -> None:
    builder.build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = _read(active.nested.base_eval.HISTORY_PATH)
    exact_events = unified_lineage_identity.build_exact_lineage_events(
        history, eval22.ASSEMBLY
    )
    requested = {
        value.strip()
        for value in os.environ.get("V23_VARIANTS", "").split(",")
        if value.strip()
    }
    unknown = requested - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown V23 variants: {sorted(unknown)}")
    selected = (
        {label: spec for label, spec in VARIANTS.items() if label in requested}
        if requested
        else VARIANTS
    )
    runs = {
        label: _run_variant(label, spec, exact_events)
        for label, spec in selected.items()
    }
    for label in VARIANTS:
        run_dir = OUTPUT_DIR / label / "active_run"
        if label not in runs and (run_dir / "summary.json").exists():
            runs[label] = run_dir

    baseline_metrics = _metrics(BASELINE_DIR)
    baseline_by = _read(BASELINE_DIR / "by_election.csv")
    baseline_lookup = baseline_by.set_index("election_id")["regional_weighted_mae_pp"]
    summary_rows = [{"variant_label": "active_v22", **baseline_metrics}]
    by_frames = [baseline_by.assign(variant_label="active_v22")]
    national_frames = [
        _read(BASELINE_DIR / "national_predictions.csv").assign(
            variant_label="active_v22"
        )
    ]
    decision_rows: list[dict[str, object]] = []

    for label, run_dir in runs.items():
        metrics = _metrics(run_dir)
        by = _read(run_dir / "by_election.csv")
        by["regional_change_vs_v22_pp"] = (
            by["regional_weighted_mae_pp"]
            - by["election_id"].map(baseline_lookup)
        )
        summary_rows.append({"variant_label": label, **metrics})
        by_frames.append(by.assign(variant_label=label))
        national_frames.append(
            _read(run_dir / "national_predictions.csv").assign(variant_label=label)
        )
        regional_change = float(
            metrics["regional_equal_election_macro_mae_pp"]
            - baseline_metrics["regional_equal_election_macro_mae_pp"]
        )
        national_change = float(
            metrics["national_equal_election_macro_mae_pp"]
            - baseline_metrics["national_equal_election_macro_mae_pp"]
        )
        max_regression = float(by["regional_change_vs_v22_pp"].max())
        gate = bool(
            metrics["winner_accuracy"] >= baseline_metrics["winner_accuracy"]
            and regional_change <= 0.10
            and national_change <= 0.10
            and max_regression <= 0.25
        )
        decision_rows.append(
            {
                "variant_label": label,
                "regional_change_vs_v22_pp": regional_change,
                "national_change_vs_v22_pp": national_change,
                "maximum_election_regression_pp": max_regression,
                "winner_accuracy": metrics["winner_accuracy"],
                "equivalence_gate": "pass" if gate else "fail",
            }
        )

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(by_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    reproduction = decisions.loc[
        decisions["variant_label"].eq("v22_reproduction")
    ].iloc[0]
    reproduction_ok = bool(
        abs(float(reproduction["regional_change_vs_v22_pp"])) <= 1e-10
        and abs(float(reproduction["national_change_vs_v22_pp"])) <= 1e-10
    )
    final_label = "v23_unified_profile_transfer_generation"
    final_gate = str(
        decisions.loc[
            decisions["variant_label"].eq(final_label), "equivalence_gate"
        ].iloc[0]
    )
    decision = {
        "experiment": "automatic_controls_v23",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "v22_reproduction_passed": reproduction_ok,
        "active_model_changed": False,
        "selection_is_development_outcome_aware": True,
        "promotion_candidate": final_label,
        "promotion_gate": final_gate if reproduction_ok else "invalid_experiment",
        "promotion_status": "not_promoted_pending_review",
        "equivalence_gate": {
            "regional_degradation_cap_pp": 0.10,
            "national_degradation_cap_pp": 0.10,
            "maximum_election_regression_cap_pp": 0.25,
            "winner_accuracy_no_regression": True,
        },
    }
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    national.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    decisions.to_csv(
        OUTPUT_DIR / "decision_table.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(decisions.to_string(index=False))
    print()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
