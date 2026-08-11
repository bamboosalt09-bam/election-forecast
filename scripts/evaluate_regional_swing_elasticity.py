"""Rolling validation of regional swing curves on direct-party elections."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import regional_swing_elasticity as swing  # noqa: E402
from scripts import evaluate_electorate_layers as electorate  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "regional_swing_elasticity_nonpresidential"
PRIOR_STRENGTHS = (2.0, 4.0, 8.0)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(electorate.HISTORY_PATH, encoding="utf-8-sig")
    events = swing.build_event_frame(history)
    event_order = (
        events[["election_id", "event_date"]]
        .drop_duplicates()
        .sort_values(["event_date", "election_id"])
    )
    rows: list[dict[str, object]] = []
    for prior_strength in PRIOR_STRENGTHS:
        for event in event_order.itertuples(index=False):
            prior_events = events.loc[events["event_date"].lt(event.event_date), "election_id"].nunique()
            if prior_events < 2:
                continue
            profiles = swing.fit_profiles(
                events, cutoff=event.event_date, prior_strength=prior_strength
            )
            target = events.loc[events["election_id"].eq(event.election_id)]
            for row in target.itertuples(index=False):
                profile = swing.profile_for_region(profiles, row.region_id)
                for method in ("flat_national", "offset", "elasticity"):
                    prediction = (
                        float(row.national_conservative_share)
                        if method == "flat_national"
                        else swing.predict_region_share(
                            profile,
                            float(row.national_conservative_share),
                            method=method,
                        )
                    )
                    rows.append(
                        {
                            "prior_strength": prior_strength,
                            "target_event": event.election_id,
                            "target_date": event.event_date,
                            "region_id": row.region_id,
                            "method": method,
                            "predicted_share": prediction,
                            "actual_share": float(row.regional_conservative_share),
                            "abs_error_pp": abs(prediction - row.regional_conservative_share) * 100.0,
                            "quality": float(row.quality),
                            "profile_source": "none" if profile is None else str(profile["source"]),
                        }
                    )
    predictions = pd.DataFrame(rows)
    summary = (
        predictions.groupby(["prior_strength", "method"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "weighted_mae_pp": float(
                        np.average(group["abs_error_pp"], weights=group["quality"])
                    ),
                    "rows": len(group),
                    "events": group["target_event"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    eligible = summary.loc[summary["method"].eq("elasticity")]
    best = eligible.loc[eligible["weighted_mae_pp"].idxmin()]
    flat = summary.loc[
        summary["prior_strength"].eq(best["prior_strength"])
        & summary["method"].eq("flat_national")
    ].iloc[0]
    offset = summary.loc[
        summary["prior_strength"].eq(best["prior_strength"])
        & summary["method"].eq("offset")
    ].iloc[0]
    payload = {
        "scope": "non_presidential_direct_party_elections_only",
        "presidential_outcomes_used": False,
        "rolling_point_in_time": True,
        "selected_prior_strength": float(best["prior_strength"]),
        "elasticity_mae_pp": float(best["weighted_mae_pp"]),
        "offset_mae_pp": float(offset["weighted_mae_pp"]),
        "flat_national_mae_pp": float(flat["weighted_mae_pp"]),
        "elasticity_change_vs_flat_pp": float(best["weighted_mae_pp"] - flat["weighted_mae_pp"]),
        "elasticity_change_vs_offset_pp": float(best["weighted_mae_pp"] - offset["weighted_mae_pp"]),
        "selected_method": "offset",
        "offset_eligible_for_presidential_ablation": bool(
            offset["weighted_mae_pp"] < flat["weighted_mae_pp"]
        ),
        "elasticity_eligible_for_presidential_ablation": bool(
            best["weighted_mae_pp"] < flat["weighted_mae_pp"]
            and best["weighted_mae_pp"] <= offset["weighted_mae_pp"]
        ),
    }
    predictions.to_csv(
        OUTPUT_DIR / "rolling_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    _atomic_json(payload, OUTPUT_DIR / "decision.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
