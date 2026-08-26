"""V32's scored panel is V31's, unchanged, and that is the point.

V32 repairs the prospective assembly: the target frame used to receive a zero
for every column it lacked, and five of those families turned out to be
model-active - two found by the initial sweep, three more only after the
contract was in place, because upstream stages create them and default them to
zero so they never entered the missing list. The
scored panel never went through that assembly, so nothing about 2002-2022
should move.

Rather than assert that, this runner produces the scored artifact by running
V31's chain into V32's directory and then requires the result to be **byte
identical** to V31's. If a single row differs, something other than the
prospective assembly changed, and the promotion stops here rather than shipping
a scored artifact nobody meant to move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import calibration_guard  # noqa: E402
from presidential_issue_engine import issue_vote_engine as engine  # noqa: E402
from presidential_issue_engine import party_regionalism_dispersion as dispersion  # noqa: E402
from presidential_issue_engine import raw_input_read_trace as read_trace  # noqa: E402
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v31 as v31  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v32"
V31_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v31"
FINAL_VARIANT = "v32_prospective_feature_contract"
V31_PREDICTION_SHA256 = "969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _traced_reader(original):
    """Record every CSV opened, and refuse the external-model-derived ones."""

    def read(path: str):
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        if name in read_trace.EXTERNAL_MODEL_DERIVED_FILENAMES:
            read_trace.record(path, reader=read_trace.REFUSED_READER)
            return pd.DataFrame()
        read_trace.record(path, reader="issue_vote_engine._read_csv_if_exists")
        return original(path)

    return read


def run(output_dir: Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)

    # The dispersion calibration alternates two constraints for up to 200
    # rounds and returns its last iterate whether or not the tolerance was met.
    # Wrapped rather than edited: the module is hash-pinned in the V30 and V31
    # manifests, so a wrapper keeps those reproducing while this version gets a
    # check. On a converged run the output is identical.
    original_calibrate = dispersion._calibrate
    original_reader = engine._read_csv_if_exists
    calibration_reports: list[dict[str, object]] = []
    try:
        dispersion._calibrate = calibration_guard.checked(
            original_calibrate, record=calibration_reports.append
        )
        # The scored path gets the same treatment the prospective path got.
        # V28 evidenced its external-model-free claim by deleting rows from the
        # manifest after the run and then reading the manifest; this records
        # what was opened, where nothing edits it, on both paths.
        engine._read_csv_if_exists = _traced_reader(original_reader)
        with read_trace.tracing() as rows:
            produced = v31.run(output_dir=destination)
            trace = read_trace.to_frame(rows)
    finally:
        dispersion._calibrate = original_calibrate
        engine._read_csv_if_exists = original_reader

    trace_path = read_trace.write(
        trace, Path(produced) / "raw_input_read_trace.csv"
    )
    read_trace.assert_no_external_model_derived_reads(
        trace, site="the V32 scored run"
    )
    predictions = Path(produced) / "nested_predictions.csv"
    digest = _sha256(predictions)
    if digest != V31_PREDICTION_SHA256:
        raise RuntimeError(
            "V32's scored panel differs from V31's. This version changes the "
            "prospective assembly only, so a scored difference means something "
            f"else was mixed in. Expected {V31_PREDICTION_SHA256}, got {digest}."
        )

    summary_path = Path(produced) / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["policy_version"] = "active_v32_prospective_feature_contract"
    payload["predecessor"] = "v31"
    payload["scored_panel_identical_to_v31"] = True
    payload["scored_panel_sha256"] = digest
    payload["prospective_feature_contract"] = {
        "changed": "the prospective target assembly only",
        "scored_panel_effect": "none; byte identical to V31 by construction and by check",
    }
    payload["metrics"]["variant"] = FINAL_VARIANT
    payload["calibration_acceptance"] = {
        "tolerance_share": calibration_guard.CALIBRATION_ABS_TOL,
        "numerical_impact_bound_pp": calibration_guard.NUMERICAL_IMPACT_BOUND_PP,
        "basis": (
            "an accuracy contract, not a figure fitted to the observed plateau: "
            "a reconciliation is accepted when it deforms a prediction by no "
            "more than a millionth of a percentage point"
        ),
        "prior_termination_condition": 1e-11,
        "prior_condition_note": (
            "the implementation reaches a residual plateau of roughly 1.9e-9 to "
            "3.8e-9 on valid input and raising the budget from 200 to 20,000 "
            "rounds does not reduce it, so the old 1e-11 condition was stricter "
            "than this implementation's numerical fixed point and the loop was "
            "in practice always exhausting its budget. Whether that plateau is a "
            "floating-point floor or a property of the alternation is not "
            "resolved here"
        ),
        "invocations": len(calibration_reports),
        "reports": calibration_reports,
    }
    payload["raw_input_read_trace"] = {
        "path": trace_path.name,
        "rows": int(len(trace)),
        "external_model_derived_reads": 0,
        "note": (
            "recorded at the point of read and never edited; refusals are kept "
            "so the record shows the engine still asks"
        ),
        # An empty trace is ambiguous on its face - it cannot be told apart
        # from instrumentation that never engaged - so the reason is stated.
        "empty_trace_reason": (
            "the scored path does not call issue_vote_engine._read_csv_if_exists "
            "at all: it reads the frozen through-2022 rederived artifacts rather "
            "than assembling from source tables, so it has no opportunity to "
            "open an external-model-derived input. Measured: 0 calls across a "
            "full scored run. The instrumentation is left installed so that a "
            "future change routing the scored path through that reader is "
            "caught rather than silently untraced"
        )
        if trace.empty
        else None,
    }
    if payload["raw_input_read_trace"]["empty_trace_reason"] is None:
        payload["raw_input_read_trace"].pop("empty_trace_reason")
    calibration_frame = pd.DataFrame(calibration_reports)
    if not calibration_frame.empty:
        v24._atomic_csv_crlf(
            calibration_frame, Path(produced) / "calibration_acceptance_audit.csv"
        )
    v24._atomic_json_crlf(payload, summary_path)
    return Path(produced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    destination = run(parser.parse_args().output_dir)
    print(v24.report(destination).to_string(index=False))


if __name__ == "__main__":
    main()
