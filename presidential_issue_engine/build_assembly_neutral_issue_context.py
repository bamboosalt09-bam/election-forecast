"""Build candidate-level neutral issue context from pre-election speech samples.

The builder does not use vote totals or vote shares. It uses candidate identity,
party, and active-slot metadata from the results registry, then compiles the
fixed 5,000-row pre-election samples into an optional active-engine input table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from presidential_issue_engine.point_in_time import forecast_cutoff  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


CONFIG_NAME = "person_party_speaker_confirmed_conf3_context050_issueglobal025_gate2"
SAMPLE_SIZE = 5_000
ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}
OUT = ROOT / "data" / "raw" / "assembly_neutral_issue_context.csv"


def _sample_path(election_id: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "assembly_stance"
        / f"pilot_{election_id}_{SAMPLE_SIZE}"
        / "review_batch.csv"
    )


def _validate_sample(election_id: str, path: Path) -> tuple[pd.DataFrame, str]:
    sample = pd.read_csv(path, low_memory=False)
    required = {"election_id", "meeting_date", "text_sha256"}
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(sample) != SAMPLE_SIZE:
        raise ValueError(f"{path} expected {SAMPLE_SIZE} rows, found {len(sample)}")
    if sample["text_sha256"].duplicated().any():
        raise ValueError(f"{path} contains duplicate text hashes")
    dates = pd.to_datetime(sample["meeting_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{path} contains missing or invalid meeting dates")
    cutoff = pd.Timestamp(forecast_cutoff(election_id, ELECTION_DATES))
    if (dates > cutoff).any():
        raise ValueError(f"{path} contains rows after forecast cutoff {cutoff.date()}")
    return sample, dates.max().date().isoformat()


def build() -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in ELECTION_DATES:
        path = _sample_path(election_id)
        sample, available_date = _validate_sample(election_id, path)
        features = build_features(config, pilot_input=path)
        current = features.loc[features["election_id"].eq(election_id)].copy()
        current = current[
            [
                "election_id",
                "slot",
                "candidate_name",
                "stance_shadow_signal",
                "evidence_count",
                "context_neutral_count",
                "context_issue_overlap_count",
                "global_context_neutral_count",
                "global_context_issue_overlap_count",
                "global_context_structure_strength",
                "global_context_content_strength",
                "global_context_strength",
                "global_context_relative_strength",
                "coverage_gate_passed",
            ]
        ].rename(columns={"stance_shadow_signal": "assembly_neutral_issue_signal"})
        current["available_date"] = available_date
        current["confidence"] = np.minimum(
            np.sqrt(pd.to_numeric(current["evidence_count"], errors="coerce").fillna(0.0)) / 4.0,
            1.0,
        ) * pd.to_numeric(current["coverage_gate_passed"], errors="coerce").fillna(0.0)
        current["sample_rows"] = len(sample)
        current["source"] = path.relative_to(ROOT).as_posix()
        current["notes"] = (
            "pre-election 5000-row stratified speech sample; confidence-power=3; "
            "target-neutral gain=0.50; issue-global gain=0.25; vote totals and shares not used by builder"
        )
        pieces.append(current)
    out = pd.concat(pieces, ignore_index=True)
    if out.duplicated(["election_id", "slot"]).any():
        raise RuntimeError("duplicate election-slot neutral context rows")
    return out.sort_values(["election_id", "slot"]).reset_index(drop=True)


def main() -> None:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows: {OUT}")


if __name__ == "__main__":
    main()
