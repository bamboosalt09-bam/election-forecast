"""Convert presidential slot Utility values to predicted vote shares."""

from __future__ import annotations

import numpy as np
import pandas as pd


def utility_to_vote_share(utilities: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """Apply softmax by election-region-model over active slots only."""

    if temperature <= 0:
        raise ValueError("softmax temperature must be greater than 0")
    frame = utilities.copy()
    frame["predicted_vote_share"] = 0.0

    group_cols = ["election_id", "region_id", "model_name"]
    for _, group in frame.groupby(group_cols, sort=False):
        active_index = group.index[group["is_active_slot"].astype(bool)]
        if len(active_index) == 0:
            continue
        values = frame.loc[active_index, "utility"].to_numpy(dtype=float) / temperature
        values = values - np.max(values)
        exp_values = np.exp(values)
        shares = exp_values / exp_values.sum()
        frame.loc[active_index, "predicted_vote_share"] = shares

    return frame[
        [
            "election_id",
            "region_id",
            "region_name",
            "province",
            "slot",
            "is_active_slot",
            "model_name",
            "utility",
            "predicted_vote_share",
        ]
    ]

