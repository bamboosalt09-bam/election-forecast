"""Rebuild V27 outside its frozen directory and verify exact predictions."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_active_presidential_model_v27  # noqa: E402


FROZEN = ROOT / "outputs/active_presidential_nested_v27/nested_predictions.csv"
FROZEN_SHA256 = "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b"
FINAL_PREDICTION_ATOL = 1e-3  # share scale: 0.10 percentage point
DIAGNOSTIC_ATOL = 1.2e-3  # share scale: 0.12 percentage point


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(FROZEN) != FROZEN_SHA256:
        raise RuntimeError(f"frozen V27 artifact drift: {sha256(FROZEN)} != {FROZEN_SHA256}")
    with tempfile.TemporaryDirectory(prefix="election_forecast_v27_") as temporary:
        destination = Path(temporary) / "active_presidential_nested_v27"
        run_active_presidential_model_v27.run(destination)
        reproduced = destination / "nested_predictions.csv"
        frozen_frame = pd.read_csv(FROZEN, low_memory=False)
        reproduced_frame = pd.read_csv(reproduced, low_memory=False)
        if list(reproduced_frame.columns) != list(frozen_frame.columns):
            raise RuntimeError("clean V27 reproduction column order differs from frozen V27")
        if reproduced_frame.shape != frozen_frame.shape:
            raise RuntimeError(
                f"clean V27 reproduction shape differs: {reproduced_frame.shape} != {frozen_frame.shape}"
            )
        numeric = list(frozen_frame.select_dtypes(include="number").columns)
        categorical = [column for column in frozen_frame.columns if column not in numeric]
        left_numeric = frozen_frame[numeric].to_numpy(dtype=float)
        right_numeric = reproduced_frame[numeric].to_numpy(dtype=float)
        finite_difference = np.abs(left_numeric - right_numeric)
        max_numeric_difference = float(np.nanmax(finite_difference))
        final_difference = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(frozen_frame["layer_pred"]).to_numpy(dtype=float)
                    - pd.to_numeric(reproduced_frame["layer_pred"]).to_numpy(dtype=float)
                )
            )
        )
        numeric_matches = np.allclose(
            left_numeric, right_numeric, rtol=0.0, atol=DIAGNOSTIC_ATOL, equal_nan=True
        )
        final_matches = final_difference <= FINAL_PREDICTION_ATOL
        if not numeric_matches or not final_matches:
            column_maxima = pd.Series(
                np.nanmax(finite_difference, axis=0), index=numeric
            ).sort_values(ascending=False)
            print("largest_numeric_column_differences:")
            print(column_maxima.head(12).to_string())
            flat_index = int(np.nanargmax(finite_difference))
            row_index, column_index = np.unravel_index(flat_index, finite_difference.shape)
            identity_columns = [
                column
                for column in ("election_id", "region_id", "candidate_name", "slot")
                if column in frozen_frame.columns
            ]
            print(f"largest_difference_row={frozen_frame.loc[row_index, identity_columns].to_dict()}")
            print(f"largest_difference_column={numeric[column_index]}")
            print(f"frozen_value={left_numeric[row_index, column_index]}")
            print(f"reproduced_value={right_numeric[row_index, column_index]}")
            raise RuntimeError(
                "clean V27 numerical reproduction exceeds tolerance: "
                f"max_abs_difference={max_numeric_difference} diagnostic_atol={DIAGNOSTIC_ATOL} "
                f"layer_pred_max_abs_difference={final_difference} "
                f"final_prediction_atol={FINAL_PREDICTION_ATOL}"
            )
        if not frozen_frame[categorical].fillna("").astype(str).equals(
            reproduced_frame[categorical].fillna("").astype(str)
        ):
            raise RuntimeError("clean V27 categorical reproduction differs from frozen V27")
        print("[clean V27 reproduction: PASS]")
        print(f"frozen_prediction_sha256={sha256(FROZEN)}")
        print(f"reproduced_byte_sha256={sha256(reproduced)}")
        print(f"max_numeric_difference={max_numeric_difference}")
        print(f"layer_pred_max_numeric_difference={final_difference}")
        print(f"diagnostic_atol={DIAGNOSTIC_ATOL}")
        print(f"final_prediction_atol={FINAL_PREDICTION_ATOL}")


if __name__ == "__main__":
    main()
