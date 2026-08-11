"""Select an untouched 5,000-row supplement from the broad corpus remainder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
SEED = "stance-v22-broad-holdout-5000-v1"
_DIRECTIONAL_CUE = re.compile(
    r"실패|무능|부패|비리|잘못|지지|찬성|환영|반대|비판|규탄|"
    r"책임|불신|신뢰|유능|유감|사과|성과|입선|일류|위법|불법"
)


def _rank(text_hash: str, namespace: str) -> str:
    return hashlib.sha256(f"{SEED}|{namespace}|{text_hash}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--used", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-v22-sha256", required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    frame = pd.read_csv(args.input.resolve(), encoding="utf-8-sig", low_memory=False).fillna("")
    used = set(
        pd.read_csv(args.used.resolve(), encoding="utf-8-sig", usecols=["text_sha256"])[
            "text_sha256"
        ].astype(str)
    )
    remainder = frame.loc[~frame["text_sha256"].astype(str).isin(used)].copy()
    remainder["selection_bucket"] = remainder["text_excerpt"].astype(str).map(
        lambda text: "cue_rich" if _DIRECTIONAL_CUE.search(text) else "general"
    )
    pieces: list[pd.DataFrame] = []
    for election in ELECTIONS:
        election_frame = remainder.loc[remainder["election_id"].eq(election)].copy()
        cue = election_frame.loc[election_frame["selection_bucket"].eq("cue_rich")].copy()
        general = election_frame.loc[election_frame["selection_bucket"].eq("general")].copy()
        cue_quota = min(800, len(cue))
        general_quota = 1_000 - cue_quota
        cue["_rank"] = cue["text_sha256"].astype(str).map(
            lambda value: _rank(value, f"{election}:cue")
        )
        general["_rank"] = general["text_sha256"].astype(str).map(
            lambda value: _rank(value, f"{election}:general")
        )
        pieces.append(cue.sort_values("_rank").head(cue_quota).drop(columns="_rank"))
        pieces.append(general.sort_values("_rank").head(general_quota).drop(columns="_rank"))
    output = pd.concat(pieces, ignore_index=True)
    if len(output) != 5_000 or output["text_sha256"].duplicated().any():
        raise RuntimeError("V22 holdout supplement is not 5,000 unique rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    state = {
        "status": "v22_broad_holdout_supplement_complete",
        "rows": len(output),
        "seed": SEED,
        "frozen_v22_sha256": args.frozen_v22_sha256,
        "content_reviewed_before_selection": False,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
        "election_counts": output["election_id"].value_counts().sort_index().to_dict(),
        "selection_bucket_counts": output["selection_bucket"].value_counts().to_dict(),
        "output": str(destination),
    }
    destination.with_suffix(".state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
