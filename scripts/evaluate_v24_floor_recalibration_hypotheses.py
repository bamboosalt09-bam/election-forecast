"""Record the pre-declared V24 core/floor recalibration sensitivity panel.

The declared primary is fixed before evaluation:

* strong incumbent-veto gain 1.00 with continuous rupture floor erosion;
* weak same-lane gain 0.50 with a theoretical 1%p floor.

The surrounding 48 variants are a sensitivity record, not an optimiser.
Realised outcomes enter only after each outcome-blind transformation completes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "outputs" / "active_presidential_nested_v24"
OUTPUT = ROOT / "outputs" / "v24_floor_recalibration_hypotheses"
STRONG_GAINS = (0.50, 1.00)
RUPTURE_EROSION_MODES = (False, True)
WEAK_GAINS = (0.25, 0.50)
WEAK_FLOOR_MODES = ("candidate_ballot", "theoretical", "none")
RECIPIENT_WEIGHT_MODES = ("affinity_only", "prediction_tilted")
THEORETICAL_FLOOR = 0.01
DECLARED_PRIMARY = {
    "strong_incumbent_veto_gain": 1.00,
    "rupture_floor_erosion_enabled": True,
    "weak_same_lane_gain": 0.50,
    "weak_floor_mode": "theoretical",
    "recipient_weight_mode": "affinity_only",
    "theoretical_floor": THEORETICAL_FLOOR,
}
FOLLOWUP_STRUCTURAL_CANDIDATE = {
    **DECLARED_PRIMARY,
    "recipient_weight_mode": "prediction_tilted",
}
FORBIDDEN_ELECTION_TOKEN = "2025"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_prospective_excluded(frame: pd.DataFrame, label: str) -> None:
    if frame["election_id"].astype(str).str.contains(FORBIDDEN_ELECTION_TOKEN).any():
        raise RuntimeError(f"{label} contains a forbidden prospective election")


def _restore_pre_extension_baseline(predictions: pd.DataFrame) -> pd.DataFrame:
    """Undo both currently recorded V24 tail extensions row by row."""

    out = predictions.copy()
    net_columns = (
        "weak_lane_refusal_transfer_net",
        "strong_veto_transfer_net",
    )
    missing = set(net_columns) - set(out.columns)
    if missing:
        raise RuntimeError(f"cannot restore baseline; missing columns: {sorted(missing)}")
    for column in net_columns:
        out["layer_pred"] -= pd.to_numeric(out[column], errors="raise")
    totals = out.groupby(["election_id", "region_id"])["layer_pred"].sum()
    if (totals - 1.0).abs().max() >= 1e-12:
        raise RuntimeError("restored baseline is not compositional")
    return out


def main() -> None:
    from presidential_issue_engine import strong_incumbent_veto
    from presidential_issue_engine import weak_same_lane_refusal
    from scripts import evaluate_preliminary_slot_shadow_nested as nested

    predictions_path = SOURCE / "nested_predictions.csv"
    predictions = pd.read_csv(
        predictions_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    _assert_prospective_excluded(predictions, "source predictions")
    baseline = _restore_pre_extension_baseline(predictions)

    summaries: list[dict[str, object]] = []
    by_election_rows: list[pd.DataFrame] = []
    national_rows: list[pd.DataFrame] = []
    transfer_rows: list[pd.DataFrame] = []
    for strong_gain in STRONG_GAINS:
        for erosion_enabled in RUPTURE_EROSION_MODES:
            after_strong, strong_audit = (
                strong_incumbent_veto.apply_strong_incumbent_veto(
                    baseline,
                    gain=strong_gain,
                    rupture_floor_erosion_enabled=erosion_enabled,
                    theoretical_floor=THEORETICAL_FLOOR,
                )
            )
            for weak_gain in WEAK_GAINS:
                for floor_mode, recipient_weight_mode in product(
                    WEAK_FLOOR_MODES,
                    RECIPIENT_WEIGHT_MODES,
                ):
                    variant = (
                        f"strong_{strong_gain:.2f}__rupture_{int(erosion_enabled)}"
                        f"__weak_{weak_gain:.2f}__floor_{floor_mode}"
                        f"__recipient_{recipient_weight_mode}"
                    )
                    transformed, weak_audit = (
                        weak_same_lane_refusal.apply_weak_same_lane_refusal(
                            after_strong,
                            gain=weak_gain,
                            floor_mode=floor_mode,
                            theoretical_floor=THEORETICAL_FLOOR,
                            recipient_weight_mode=recipient_weight_mode,
                        )
                    )
                    _assert_prospective_excluded(transformed, variant)
                    totals = transformed.groupby(["election_id", "region_id"])[
                        "layer_pred"
                    ].sum()
                    if (totals - 1.0).abs().max() >= 1e-12:
                        raise RuntimeError(f"{variant} is not compositional")

                    # Outcomes enter only here, after transformation.
                    summary, by_election, national = nested._metrics(
                        transformed,
                        "layer_pred",
                        variant,
                    )
                    declared_primary = bool(
                        strong_gain
                        == DECLARED_PRIMARY["strong_incumbent_veto_gain"]
                        and erosion_enabled
                        == DECLARED_PRIMARY["rupture_floor_erosion_enabled"]
                        and weak_gain == DECLARED_PRIMARY["weak_same_lane_gain"]
                        and floor_mode == DECLARED_PRIMARY["weak_floor_mode"]
                        and recipient_weight_mode
                        == DECLARED_PRIMARY["recipient_weight_mode"]
                    )
                    followup_candidate = bool(
                        strong_gain
                        == FOLLOWUP_STRUCTURAL_CANDIDATE[
                            "strong_incumbent_veto_gain"
                        ]
                        and erosion_enabled
                        == FOLLOWUP_STRUCTURAL_CANDIDATE[
                            "rupture_floor_erosion_enabled"
                        ]
                        and weak_gain
                        == FOLLOWUP_STRUCTURAL_CANDIDATE["weak_same_lane_gain"]
                        and floor_mode
                        == FOLLOWUP_STRUCTURAL_CANDIDATE["weak_floor_mode"]
                        and recipient_weight_mode
                        == FOLLOWUP_STRUCTURAL_CANDIDATE[
                            "recipient_weight_mode"
                        ]
                    )
                    summary.update(
                        {
                            "strong_incumbent_veto_gain": strong_gain,
                            "rupture_floor_erosion_enabled": erosion_enabled,
                            "weak_same_lane_gain": weak_gain,
                            "weak_floor_mode": floor_mode,
                            "recipient_weight_mode": recipient_weight_mode,
                            "theoretical_floor": THEORETICAL_FLOOR,
                            "regional_row_macro_mae_pp": float(
                                (transformed["layer_pred"] - transformed["actual"])
                                .abs()
                                .mean()
                                * 100.0
                            ),
                            "declared_primary": declared_primary,
                            "followup_structural_candidate": followup_candidate,
                        }
                    )
                    summaries.append(summary)
                    by_election.insert(1, "strong_incumbent_veto_gain", strong_gain)
                    by_election.insert(
                        2,
                        "rupture_floor_erosion_enabled",
                        erosion_enabled,
                    )
                    by_election.insert(3, "weak_same_lane_gain", weak_gain)
                    by_election.insert(4, "weak_floor_mode", floor_mode)
                    by_election.insert(
                        5,
                        "recipient_weight_mode",
                        recipient_weight_mode,
                    )
                    by_election_rows.append(by_election)
                    national.insert(1, "strong_incumbent_veto_gain", strong_gain)
                    national.insert(
                        2,
                        "rupture_floor_erosion_enabled",
                        erosion_enabled,
                    )
                    national.insert(3, "weak_same_lane_gain", weak_gain)
                    national.insert(4, "weak_floor_mode", floor_mode)
                    national.insert(
                        5,
                        "recipient_weight_mode",
                        recipient_weight_mode,
                    )
                    national_rows.append(national)

                    for layer, audit in (
                        ("strong_incumbent_veto", strong_audit),
                        ("weak_same_lane_refusal", weak_audit),
                    ):
                        if audit.empty:
                            continue
                        recorded = audit.copy()
                        recorded.insert(0, "variant", variant)
                        recorded.insert(1, "layer", layer)
                        transfer_rows.append(recorded)

    metrics = pd.DataFrame(summaries).sort_values(
        [
            "strong_incumbent_veto_gain",
            "rupture_floor_erosion_enabled",
            "weak_same_lane_gain",
            "weak_floor_mode",
            "recipient_weight_mode",
        ]
    )
    by_election = pd.concat(by_election_rows, ignore_index=True)
    national = pd.concat(national_rows, ignore_index=True)
    transfers = pd.concat(transfer_rows, ignore_index=True, sort=False)
    _assert_prospective_excluded(by_election, "by-election sensitivity")
    _assert_prospective_excluded(national, "national sensitivity")
    _assert_prospective_excluded(transfers, "transfer sensitivity")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT / "by_election.csv",
        index=False,
        encoding="utf-8-sig",
    )
    national.to_csv(
        OUTPUT / "national_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transfers.to_csv(
        OUTPUT / "transfer_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "floor sensitivity record, not coefficient optimisation",
        "selection_policy": (
            "The primary combination was declared before evaluation: a full "
            "rejection-strength gain, continuous rupture-only floor erosion, "
            "and half of weak third-candidate mass above a theoretical 1%p floor."
        ),
        "declared_primary": DECLARED_PRIMARY,
        "followup_structural_candidate": FOLLOWUP_STRUCTURAL_CANDIDATE,
        "followup_reason": (
            "The original affinity-only candidate failed the existing winner "
            "safety gate because it assigned 100% of removed C mass to one major. "
            "The follow-up uses forecast-share priors tilted by the existing "
            "affinity power, so neither major receives structural zero weight."
        ),
        "strong_gain_grid": list(STRONG_GAINS),
        "rupture_erosion_modes": list(RUPTURE_EROSION_MODES),
        "weak_gain_grid": list(WEAK_GAINS),
        "weak_floor_modes": list(WEAK_FLOOR_MODES),
        "recipient_weight_modes": list(RECIPIENT_WEIGHT_MODES),
        "transformation_fields_not_read": [
            "actual",
            "votes",
            "vote_share",
            "realised_margin",
            "polling",
        ],
        "evaluation_only_fields": ["actual", "contest_votes"],
        "prospective_elections_excluded": True,
        "forbidden_election_token": FORBIDDEN_ELECTION_TOKEN,
        "source_predictions": str(predictions_path.relative_to(ROOT)),
        "source_predictions_sha256": _sha256(predictions_path),
        "baseline_reconstruction": (
            "Subtract the two recorded row-level extension net columns from "
            "the current public forecast; retain the lineage ceiling."
        ),
        "variant_count": int(len(metrics)),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
