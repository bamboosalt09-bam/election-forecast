"""Bounded preservation of strictly prior direct-party regional terrain."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def apply_direct_party_center(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    gain_by_election: Mapping[str, float],
) -> pd.DataFrame:
    """Blend eligible major-party mass toward its prior party-ballot split.

    The total share held by ineligible minor-party or independent candidates is
    preserved exactly. Within the eligible pool, prior direct-party shares are
    used only as a bounded center; candidate and issue layers retain the rest.
    """

    required = {
        "election_id",
        "region_id",
        "major_party_core_eligible",
        "direct_party_recent_base",
        prediction_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"direct-party center frame missing columns: {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["direct_party_center_gain"] = 0.0
    out["direct_party_center_target"] = out[prediction_column].astype(float)
    out["direct_party_center_shift"] = 0.0

    for (election_id, _), indices in out.groupby(
        ["election_id", "region_id"], sort=False
    ).indices.items():
        idx = np.asarray(indices, dtype=int)
        prediction = pd.to_numeric(
            out.loc[idx, prediction_column], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        eligible = out.loc[idx, "major_party_core_eligible"].fillna(False).astype(bool).to_numpy(copy=True)
        direct = pd.to_numeric(
            out.loc[idx, "direct_party_recent_base"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        eligible &= direct > 0.0
        gain = float(np.clip(gain_by_election.get(str(election_id), 0.0), 0.0, 0.35))
        if gain <= 0.0 or eligible.sum() < 2 or float(direct[eligible].sum()) <= 0.0:
            continue
        pool = float(prediction[eligible].sum())
        target = prediction.copy()
        target[eligible] = pool * direct[eligible] / float(direct[eligible].sum())
        adjusted = prediction.copy()
        adjusted[eligible] = (
            (1.0 - gain) * prediction[eligible] + gain * target[eligible]
        )
        if not np.isclose(adjusted.sum(), prediction.sum(), atol=1e-12):
            raise RuntimeError("direct-party center failed vote-mass conservation")
        out.loc[idx, "direct_party_center_gain"] = gain
        out.loc[idx, "direct_party_center_target"] = target
        out.loc[idx, "direct_party_center_shift"] = adjusted - prediction
        out.loc[idx, prediction_column] = adjusted
    return out
