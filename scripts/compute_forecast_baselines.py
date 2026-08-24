"""Compute transparent regional baselines for the frozen V23 forecasts.

The uniform-national-swing baseline is deliberately given the realized national
vote shares for the target election. It is therefore an oracle-aided diagnostic,
not a deployable forecast. Historical nationwide shares are equal-region means
because ``bloc_history_results.csv`` contains shares, not regional vote counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "active_presidential_nested_v23" / "nested_predictions.csv"
BLOC_HISTORY = ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv"
REGIONS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "regions_master.csv"
OUTPUT_DIR = ROOT / "outputs" / "forecast_baselines"

PREDICTION_REQUIRED = {
    "election_id",
    "region_id",
    "bloc",
    "contest_votes",
    "actual",
    "layer_pred",
}
HISTORY_REQUIRED = {"election_id", "election_type", "region_id", "bloc", "vote_share"}


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS, encoding="utf-8-sig")
    history = pd.read_csv(BLOC_HISTORY, encoding="utf-8-sig")
    regions = pd.read_csv(REGIONS, encoding="utf-8-sig")

    missing_predictions = PREDICTION_REQUIRED - set(predictions.columns)
    missing_history = HISTORY_REQUIRED - set(history.columns)
    if missing_predictions:
        raise ValueError(f"nested predictions missing columns: {sorted(missing_predictions)}")
    if missing_history:
        raise ValueError(f"bloc history missing columns: {sorted(missing_history)}")
    if not {"region_id", "region_name"}.issubset(regions.columns):
        raise ValueError("regions_master.csv must contain region_id and region_name")

    unknown_regions = sorted(set(predictions["region_id"]) - set(regions["region_id"]))
    if unknown_regions:
        raise ValueError(f"nested predictions contain unknown regions: {unknown_regions}")

    predictions = predictions.copy()
    for column in ("contest_votes", "actual", "layer_pred"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    if predictions["contest_votes"].le(0).any():
        raise ValueError("contest_votes must be positive")

    history = history.loc[history["election_type"].eq("presidential")].copy()
    history["vote_share"] = pd.to_numeric(history["vote_share"], errors="raise")
    history = (
        history.groupby(["election_id", "region_id", "bloc"], as_index=False)["vote_share"]
        .sum()
        .sort_values(["election_id", "region_id", "bloc"])
    )
    return predictions, history, regions


def _election_year(election_id: str) -> int:
    try:
        return int(election_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unsupported election_id: {election_id}") from exc


def _previous_election(election_id: str, available: list[str]) -> str:
    target_year = _election_year(election_id)
    prior = [item for item in available if _election_year(item) < target_year]
    if not prior:
        raise ValueError(f"no prior presidential election for {election_id}")
    return max(prior, key=_election_year)


def _renormalize(frame: pd.DataFrame, value_column: str) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="raise").clip(lower=0.0)
    totals = values.groupby(frame["region_id"]).transform("sum")
    if totals.le(0).any():
        bad = sorted(frame.loc[totals.le(0), "region_id"].unique())
        raise ValueError(f"cannot renormalize zero-sum regions: {bad}")
    return values / totals


def _weighted_mae_pp(frame: pd.DataFrame, prediction_column: str) -> float:
    error_pp = (frame[prediction_column] - frame["actual"]).abs() * 100.0
    return float(np.average(error_pp, weights=frame["contest_votes"]))


def _build_election_baselines(
    target: pd.DataFrame,
    history: pd.DataFrame,
    previous_id: str,
) -> pd.DataFrame:
    previous = history.loc[history["election_id"].eq(previous_id), ["region_id", "bloc", "vote_share"]]
    frame = target.merge(previous, on=["region_id", "bloc"], how="left")
    frame["has_prior_bloc"] = frame.groupby("bloc")["vote_share"].transform(
        lambda values: values.notna().any()
    )
    frame["prior_share"] = frame["vote_share"].fillna(0.0)

    # A newly created region (Sejong in the 2012 fold) has no previous regional
    # observation. A neutral within-region distribution avoids importing a
    # later region while keeping the denominator well-defined.
    prior_totals = frame.groupby("region_id")["prior_share"].transform("sum")
    empty_region = prior_totals.le(0.0)
    if empty_region.any():
        group_sizes = frame.loc[empty_region].groupby("region_id")["prior_share"].transform("size")
        frame.loc[empty_region, "prior_share"] = 1.0 / group_sizes

    frame["persistence"] = _renormalize(frame, "prior_share")

    # This current national result is intentionally supplied to make UNS an
    # oracle-aided and therefore favorable comparator. It must not be treated
    # as a deployable forecast baseline.
    current_national = frame.groupby("bloc", sort=False).apply(
        lambda group: np.average(group["actual"], weights=group["contest_votes"]),
        include_groups=False,
    )
    previous_national = (
        history.loc[history["election_id"].eq(previous_id)]
        .groupby("bloc", sort=False)["vote_share"]
        .mean()
    )
    swing = current_national.subtract(previous_national, fill_value=np.nan)
    frame["swing"] = frame["bloc"].map(swing).fillna(0.0)
    frame.loc[~frame["has_prior_bloc"], "swing"] = 0.0
    frame["uniform_national_swing_raw"] = frame["prior_share"] + frame["swing"]
    frame["uniform_national_swing"] = _renormalize(frame, "uniform_national_swing_raw")

    frame["national_uniform"] = frame["bloc"].map(current_national)
    frame["national_uniform"] = _renormalize(frame, "national_uniform")
    return frame


def compute_baselines(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    available = sorted(history["election_id"].unique(), key=_election_year)
    rows: list[dict[str, object]] = []
    for election_id, target in predictions.groupby("election_id", sort=True):
        previous_id = _previous_election(str(election_id), available)
        frame = _build_election_baselines(target.copy(), history, previous_id)
        model_mae = _weighted_mae_pp(frame, "layer_pred")
        persistence_mae = _weighted_mae_pp(frame, "persistence")
        swing_mae = _weighted_mae_pp(frame, "uniform_national_swing")
        uniform_mae = _weighted_mae_pp(frame, "national_uniform")
        rows.append(
            {
                "election_id": election_id,
                "previous_election_id": previous_id,
                "model_mae_pp": model_mae,
                "persistence_mae_pp": persistence_mae,
                "uniform_national_swing_mae_pp": swing_mae,
                "national_uniform_mae_pp": uniform_mae,
                "skill_vs_persistence": 1.0 - model_mae / persistence_mae,
                "skill_vs_uniform_national_swing": 1.0 - model_mae / swing_mae,
            }
        )
    return pd.DataFrame(rows)


def _skill_test(values: pd.Series) -> dict[str, float | int]:
    sample = pd.to_numeric(values, errors="raise").to_numpy(float)
    n = int(sample.size)
    mean = float(np.mean(sample))
    sd = float(np.std(sample, ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "t_statistic": float("nan"),
            "p_value_two_sided": float("nan"),
            "ci95_low": mean,
            "ci95_high": mean,
        }
    standard_error = sd / np.sqrt(n)
    t_statistic = mean / standard_error
    critical = float(stats.t.ppf(0.975, df=n - 1))
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "t_statistic": float(t_statistic),
        "p_value_two_sided": p_value,
        "ci95_low": float(mean - critical * standard_error),
        "ci95_high": float(mean + critical * standard_error),
    }


def build_summary(by_election: pd.DataFrame) -> dict[str, object]:
    metric_columns = (
        "model_mae_pp",
        "persistence_mae_pp",
        "uniform_national_swing_mae_pp",
        "national_uniform_mae_pp",
    )
    # The filename is generic and the directory is not versioned, so the
    # artifact has to say which model it measured. It said so only because
    # someone added the two fields by hand, and this script rebuilds the file
    # from scratch - the next re-run silently dropped them. Derived here, they
    # survive re-running.
    measured = PREDICTIONS.parent.name.rsplit("_", 1)[-1]
    return {
        "schema": "forecast_baseline_summary_v1",
        "model_version": measured,
        "model_version_note": (
            f"These baselines were computed against the {measured.upper()} active "
            "model and have not been recomputed since; the generic filename does "
            "not imply the current active version. See "
            "data/config/current_presidential_model.json."
        ),
        "elections": int(len(by_election)),
        "weighting": "contest_votes within election; equal weight across elections",
        "uniform_national_swing_is_oracle_aided": True,
        "macro_mae_pp": {
            column.removesuffix("_mae_pp"): float(by_election[column].mean())
            for column in metric_columns
        },
        "skill_tests": {
            "vs_persistence": _skill_test(by_election["skill_vs_persistence"]),
            "vs_uniform_national_swing": _skill_test(
                by_election["skill_vs_uniform_national_swing"]
            ),
        },
    }


def main() -> None:
    predictions, history, _ = _read_inputs()
    by_election = compute_baselines(predictions, history)
    summary = build_summary(by_election)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_election.to_csv(OUTPUT_DIR / "baseline_by_election.csv", index=False, encoding="utf-8-sig")
    # write_bytes, not write_text: on Windows the latter translates LF to
    # CRLF and rewrites every line of a file nothing meant to change
    (OUTPUT_DIR / "baseline_summary.json").write_bytes(
        (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    print(by_election.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
