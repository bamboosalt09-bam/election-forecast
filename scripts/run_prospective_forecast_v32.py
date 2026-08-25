"""V32's 2025 forecast: build the active features instead of zeroing them.

The prospective assembly closed the gap between the historical frame and the
target frame by setting every missing column to zero, under a comment asserting
such columns were diagnostic-only. Two families it caught were model-active:

* ``regional_accent_*`` - 27 columns feeding the accent gain, which gates the
  log shift, which moves the prediction. The published 2025 forecast ran with
  that layer contributing exactly nothing, while the scored elections carry
  gains of 0.10 to 0.20.
* ``major_party_core_eligible`` - every 2025 candidate was marked ineligible,
  including both major-party nominees, so their durable core was zeroed and
  folded into critical support.

Neither shows in the output, because zero is a legal value everywhere it
landed.

This runner installs ``prospective_feature_contract.resolve`` in place of the
zero-fill and supplies a builder for each family. The accent is computed from
history strictly before the 2025-06-02 cutoff, by the same functions the scored
elections use; eligibility is the same bloc membership test. Anything missing
that is neither an outcome, a declared default, nor a family with a builder
stops the run.

The correctness argument does not depend on the result. Discarding a computable
active input is wrong whether or not the 2025 score moves, and this version is
promoted either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
from presidential_issue_engine import prospective_feature_contract as contract  # noqa: E402
from scripts import run_prospective_forecast as prospective  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402
from scripts import run_prospective_forecast_v31 as v31  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "prospective_pres_2025_v32"
TARGET_ELECTION = "pres_2025"
#: the profile prepare_frame uses for the scored rows
MASS_PROFILE = "direct_party_layers"
#: The candidate conversion context that includes the forecast target.
CONVERSION_CONTEXT_WITH_TARGET = prospective.CANDIDATE_CONVERSION_HISTORY


def _history() -> pd.DataFrame:
    source = prospective.HISTORY
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(source, encoding="utf-8-sig")


def _canonical_layers(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the target rows through the same estimator the scored rows get.

    An earlier version of this builder recomputed the accent by calling
    ``_attach_candidate_regional_accent`` directly on the target frame. That
    produced values differing from the canonical ones by up to 0.036 in
    reliability, because inside the estimator that attachment runs after
    ``_candidate_camp_frame`` and the rest of the layer assembly. Calling the
    estimator is the only way to be equal to it by construction rather than by
    resemblance.
    """

    keys = ["election_id", "region_id", "slot", "bloc"]
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise contract.ProspectiveFeatureError(f"target frame lacks layer keys {missing}")
    landscape = [c for c in frame.columns if c.startswith("landscape_")]
    candidate_keys = frame[keys + landscape].copy()
    estimated = layers.estimate_electorate_layers(
        candidate_keys, _history(), mass_profile=MASS_PROFILE
    )
    if len(estimated) != len(frame):
        raise contract.ProspectiveFeatureError(
            f"the layer estimator returned {len(estimated)} rows for {len(frame)} target rows"
        )
    estimated.index = frame.index
    return estimated


def _build_regional_accent(frame: pd.DataFrame) -> pd.DataFrame:
    return _canonical_layers(frame)


def _eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    return _canonical_layers(frame)


BUILDERS = {
    "regional_accent": _build_regional_accent,
    "major_party_core_eligible": _eligibility,
}


def _contract(frame: pd.DataFrame, historical_columns, *, site: str) -> pd.DataFrame:
    return contract.resolve(frame, historical_columns, BUILDERS, site=site)


def run() -> Path:
    original_contract = prospective.TARGET_FEATURE_CONTRACT
    original_output = v31.OUTPUT_DIR
    original_conversion = active.CONVERSION_CONTEXT
    try:
        prospective.TARGET_FEATURE_CONTRACT = _contract
        # The strategic-lane consumer reads this module constant directly, and
        # it points at the history-only table. The prospective path already
        # prepares a context that carries the target - available 2025-06-02,
        # the D-1 cutoff - so the merge finds nothing only because the consumer
        # is looking somewhere else. wasted_vote_resistance and
        # strategic_transfer_confidence were zeroed for exactly this reason.
        active.CONVERSION_CONTEXT = CONVERSION_CONTEXT_WITH_TARGET
        v31.OUTPUT_DIR = OUTPUT_DIR
        destination = v31.run()
    finally:
        prospective.TARGET_FEATURE_CONTRACT = original_contract
        active.CONVERSION_CONTEXT = original_conversion
        v31.OUTPUT_DIR = original_output

    manifest_path = Path(destination) / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["version"] = "v32"
    payload["prospective_feature_contract"] = {
        "replaces": "blanket zero-fill of every column the target lacked",
        "outcome_only_set_to_nan": sorted(contract.OUTCOME_COLUMNS),
        "explicit_zero": dict(contract.EXPLICIT_ZERO_COLUMNS),
        "diagnostic_only": dict(contract.DIAGNOSTIC_ONLY_COLUMNS),
        "required_derived_families_built": sorted(BUILDERS),
        "unclassified_missing_column_behaviour": "raise",
        "target_election_outcome_fields_used": [],
    }
    manifest_path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return Path(destination)


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
