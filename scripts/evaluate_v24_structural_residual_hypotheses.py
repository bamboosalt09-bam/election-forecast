"""Record sensitivity for two narrow, outcome-blind V24 hypotheses.

This evaluator does not select coefficients.  Its declared primary pair is a
round doubling of the incumbent-veto gain (0.50) and a quarter of the weak
third-candidate excess (0.25).  The surrounding grid is retained only to show
sensitivity.  Realised outcomes are passed solely to the existing metric
function after each transformation has finished.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "outputs" / "active_presidential_nested_v24"
OUTPUT = ROOT / "outputs" / "v24_structural_residual_hypotheses"
STRONG_GAINS = (0.00, 0.25, 0.50, 1.00)
WEAK_GAINS = (0.00, 0.10, 0.25, 0.50)
DECLARED_PRIMARY = {"strong_incumbent_veto_gain": 0.50, "weak_same_lane_gain": 0.25}
FORBIDDEN_ELECTION_TOKEN = "2025"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_prospective_excluded(frame: pd.DataFrame, label: str) -> None:
    if frame["election_id"].astype(str).str.contains(FORBIDDEN_ELECTION_TOKEN).any():
        raise RuntimeError(f"{label} contains a forbidden prospective election")


def _undo_current_extensions(
    predictions: pd.DataFrame,
    strong_audit: pd.DataFrame,
    weak_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Recover the routing-plus-lineage frame from the current public output."""

    out = predictions.copy()
    generated_net_columns = (
        "weak_lane_refusal_transfer_net",
        "strong_veto_transfer_net",
    )
    missing = set(generated_net_columns) - set(out.columns)
    if missing:
        raise RuntimeError(f"cannot restore baseline; missing net columns: {sorted(missing)}")
    if strong_audit.empty or weak_audit.empty:
        raise RuntimeError("current extension audits must be non-empty")
    # Reverse the exact recorded row-level net changes. The two transformations
    # are mass-preserving, and their audit columns travel with the final frame.
    for column in generated_net_columns:
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
    strong_audit_path = SOURCE / "strong_incumbent_veto_audit.csv"
    weak_audit_path = SOURCE / "weak_same_lane_refusal_audit.csv"
    predictions = pd.read_csv(
        predictions_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    current_strong_audit = pd.read_csv(strong_audit_path, encoding="utf-8-sig")
    current_weak_audit = pd.read_csv(weak_audit_path, encoding="utf-8-sig")
    _assert_prospective_excluded(predictions, "source predictions")
    _assert_prospective_excluded(current_strong_audit, "source strong-veto audit")
    _assert_prospective_excluded(current_weak_audit, "source weak-refusal audit")
    baseline = _undo_current_extensions(
        predictions,
        current_strong_audit,
        current_weak_audit,
    )

    summaries: list[dict[str, object]] = []
    by_election_rows: list[pd.DataFrame] = []
    transfer_rows: list[pd.DataFrame] = []
    for strong_gain in STRONG_GAINS:
        after_strong, strong_audit = strong_incumbent_veto.apply_strong_incumbent_veto(
            baseline,
            gain=strong_gain,
            rupture_floor_erosion_enabled=False,
        )
        for weak_gain in WEAK_GAINS:
            variant = f"strong_{strong_gain:.2f}__weak_{weak_gain:.2f}"
            transformed, weak_audit = (
                weak_same_lane_refusal.apply_weak_same_lane_refusal(
                    after_strong,
                    gain=weak_gain,
                    floor_mode="candidate_ballot",
                    recipient_weight_mode="affinity_only",
                )
            )
            _assert_prospective_excluded(transformed, variant)
            totals = transformed.groupby(["election_id", "region_id"])[
                "layer_pred"
            ].sum()
            if (totals - 1.0).abs().max() >= 1e-12:
                raise RuntimeError(f"{variant} is not compositional")

            # Outcomes enter for the first time here, in evaluation only.
            summary, by_election, _ = nested._metrics(
                transformed,
                "layer_pred",
                variant,
            )
            summary.update(
                {
                    "strong_incumbent_veto_gain": strong_gain,
                    "weak_same_lane_gain": weak_gain,
                    "regional_row_macro_mae_pp": float(
                        (transformed["layer_pred"] - transformed["actual"])
                        .abs()
                        .mean()
                        * 100.0
                    ),
                    "declared_primary": bool(
                        strong_gain
                        == DECLARED_PRIMARY["strong_incumbent_veto_gain"]
                        and weak_gain == DECLARED_PRIMARY["weak_same_lane_gain"]
                    ),
                }
            )
            summaries.append(summary)
            by_election.insert(1, "strong_incumbent_veto_gain", strong_gain)
            by_election.insert(2, "weak_same_lane_gain", weak_gain)
            by_election_rows.append(by_election)

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
        ["strong_incumbent_veto_gain", "weak_same_lane_gain"]
    )
    by_election = pd.concat(by_election_rows, ignore_index=True)
    transfers = (
        pd.concat(transfer_rows, ignore_index=True, sort=False)
        if transfer_rows
        else pd.DataFrame()
    )
    _assert_prospective_excluded(by_election, "by-election sensitivity")
    if not transfers.empty:
        _assert_prospective_excluded(transfers, "transfer sensitivity")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT / "by_election.csv",
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
        "purpose": "sensitivity record, not coefficient optimisation",
        "selection_policy": (
            "The declared primary gains were fixed before this grid was evaluated: "
            "0.50 is a round doubling of 0.25 and 0.25 moves one quarter of the "
            "weak candidate excess. Historical metrics did not choose either value."
        ),
        "declared_primary": DECLARED_PRIMARY,
        "strong_gain_grid": list(STRONG_GAINS),
        "weak_gain_grid": list(WEAK_GAINS),
        "transformation_fields_read": [
            "election_id",
            "region_id",
            "source_slot",
            "layer_pred",
            "government_direction_score",
            "government_rejection_strength",
            "dominant_slot",
            "runner_up_slot",
            "dominance_activation",
            "regime_certainty",
            "regime_core_floor",
            "candidate_ballot_recent_base",
            "landscape_axis_*",
            "third_candidate_lineage pre-election fields",
        ],
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
        "source_strong_audit": str(strong_audit_path.relative_to(ROOT)),
        "source_strong_audit_sha256": _sha256(strong_audit_path),
        "source_weak_audit": str(weak_audit_path.relative_to(ROOT)),
        "source_weak_audit_sha256": _sha256(weak_audit_path),
        "baseline_reconstruction": (
            "Subtract both recorded row-level extension net columns from the "
            "current public forecast. The existing lineage ceiling is retained."
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
