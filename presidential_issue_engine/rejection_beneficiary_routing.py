"""Route cumulative incumbent-rejection mass within the major-party contest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.contest_regime import conservative_core_floor


def apply_rejection_beneficiary_routing(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    prediction_column: str,
    slot_column: str = "source_slot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move rejected runner-up flexible mass to the dominant major candidate.

    The transfer rate is the already reliability-discounted cumulative
    rejection advantage. No election-specific gain is introduced. Third-party
    support is unchanged, and the rejected candidate's conservative core floor
    cannot be crossed.
    """

    required = {"election_id", "region_id", slot_column, prediction_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rejection routing frame missing columns: {missing}")
    regime_required = {
        "election_id",
        "dominant_slot",
        "runner_up_slot",
        "regime_rejection_activation",
        "regime_certainty",
        "cumulative_rejection_advantage",
    }
    if regimes.empty or not regime_required.issubset(regimes.columns):
        return frame.copy(), pd.DataFrame()

    out = frame.copy().reset_index(drop=True)
    out["rejection_beneficiary_transfer_in"] = 0.0
    out["rejection_beneficiary_transfer_out"] = 0.0
    out["rejection_beneficiary_transfer_net"] = 0.0
    out["rejection_beneficiary_rate"] = 0.0
    out["rejection_beneficiary_core_floor"] = conservative_core_floor(out)
    regime_lookup = regimes.drop_duplicates("election_id").set_index("election_id")
    values = (
        pd.to_numeric(out[prediction_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        .copy()
    )
    audit_rows: list[dict[str, object]] = []

    for (election_id, region_id), group in out.groupby(
        ["election_id", "region_id"], sort=False
    ):
        if election_id not in regime_lookup.index:
            continue
        regime = regime_lookup.loc[election_id]
        activation = float(
            np.clip(regime["regime_rejection_activation"], 0.0, 1.0)
        )
        certainty = float(np.clip(regime["regime_certainty"], 0.0, 1.0))
        advantage = float(
            np.clip(regime["cumulative_rejection_advantage"], 0.0, 1.0)
        )
        rate = activation * certainty * advantage
        if rate <= 0.0:
            continue

        dominant_slot = str(regime["dominant_slot"])
        runner_slot = str(regime["runner_up_slot"])
        dominant_index = group.index[
            group[slot_column].astype(str).eq(dominant_slot)
        ]
        runner_index = group.index[group[slot_column].astype(str).eq(runner_slot)]
        if len(dominant_index) != 1 or len(runner_index) != 1:
            continue
        dominant_index = int(dominant_index[0])
        runner_index = int(runner_index[0])
        if "major_party_core_eligible" in out.columns:
            eligible = out.loc[
                [dominant_index, runner_index], "major_party_core_eligible"
            ].fillna(False).astype(bool)
            if not bool(eligible.all()):
                continue

        runner_floor = float(
            np.clip(out.at[runner_index, "rejection_beneficiary_core_floor"], 0.0, 1.0)
        )
        flexible = max(values[runner_index] - runner_floor, 0.0)
        transfer = min(rate * flexible, values[runner_index])
        if transfer <= 0.0:
            continue
        values[runner_index] -= transfer
        values[dominant_index] += transfer
        out.at[runner_index, "rejection_beneficiary_transfer_out"] = transfer
        out.at[dominant_index, "rejection_beneficiary_transfer_in"] = transfer
        out.loc[[dominant_index, runner_index], "rejection_beneficiary_rate"] = rate
        audit_rows.append(
            {
                "election_id": str(election_id),
                "region_id": str(region_id),
                "dominant_slot": dominant_slot,
                "runner_up_slot": runner_slot,
                "rejection_rate": rate,
                "runner_core_floor": runner_floor,
                "runner_flexible_mass": flexible,
                "transfer": transfer,
            }
        )

    out["rejection_beneficiary_transfer_net"] = (
        out["rejection_beneficiary_transfer_in"]
        - out["rejection_beneficiary_transfer_out"]
    )
    out[prediction_column] = values
    return out, pd.DataFrame(audit_rows)
