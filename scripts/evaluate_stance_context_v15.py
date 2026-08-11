"""Evaluate the frozen NLI classifier with the V15 ownership gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from election_forecast.stance_context_v15 import (  # noqa: E402
    apply_contextual_ownership_gate_v15,
)
from election_forecast.stance_precision import (  # noqa: E402
    apply_ambiguity_abstention,
    precision_first_metrics,
)
from scripts.apply_stance_precision_ensemble import validate_shadow_corpus  # noqa: E402


MODEL_DIR = ROOT / "outputs" / "assembly_stance" / "stance_ko_nli_context_v10"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_ownership_v15"
ARTIFACT = MODEL_DIR / "stance_ko_nli_context_v10.joblib"
APPLICATIONS = {
    "application_5000": MODEL_DIR / "application_5000" / "context_predictions_5000.csv",
    "application_unseen_5000": MODEL_DIR
    / "application_unseen_5000"
    / "context_predictions_5000.csv",
    "application_unseen_followup_4000": MODEL_DIR
    / "application_unseen_followup_4000"
    / "context_predictions_5000.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audit(version: int, predictions: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(
        ROOT / "data" / "shadow" / f"stance_locked_audit_v{version}.csv",
        encoding="utf-8-sig",
    ).fillna("")
    audit = audit.drop(
        columns=[
            "audit_locked_label",
            "audit_target_correct",
            "audit_quotation_owner",
            "audit_notes",
            "context_prediction",
            "ambiguity_gated_prediction",
        ],
        errors="ignore",
    )
    labels = pd.read_csv(
        ROOT / "data" / "shadow" / f"stance_locked_audit_v{version}_labels.csv",
        encoding="utf-8-sig",
    ).fillna("")
    return audit.merge(labels, on="text_sha256", validate="one_to_one").merge(
        predictions[["text_sha256", "context_prediction"]],
        on="text_sha256",
        validate="one_to_one",
    )


def _metrics(frame: pd.DataFrame, prediction: pd.Series) -> dict[str, float | int]:
    truth = frame["audit_locked_label"].where(
        frame["audit_target_correct"].astype(str).str.lower().eq("true"),
        "neutral",
    )
    return precision_first_metrics(truth, prediction)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pieces: list[pd.DataFrame] = []
    application_summary: dict[str, object] = {}
    for name, path in APPLICATIONS.items():
        frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        validate_shadow_corpus(frame)
        v14, v14_reasons = apply_ambiguity_abstention(
            frame.to_dict(orient="records"), frame["context_prediction"]
        )
        v15, v15_reasons = apply_contextual_ownership_gate_v15(
            frame.to_dict(orient="records"), frame["context_prediction"]
        )
        output = frame.copy()
        output["v14_prediction"] = v14
        output["v14_abstention_reasons"] = v14_reasons
        output["v15_prediction"] = v15
        output["v15_abstention_reasons"] = v15_reasons
        destination = OUTPUT_DIR / name / "context_predictions_v15.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(destination, index=False, encoding="utf-8-sig")
        pieces.append(output)
        changed = output["v14_prediction"].ne(output["v15_prediction"])
        application_summary[name] = {
            "rows": int(len(output)),
            "v14_directional": int(output["v14_prediction"].ne("neutral").sum()),
            "v15_directional": int(output["v15_prediction"].ne("neutral").sum()),
            "v15_additional_abstentions": int(changed.sum()),
            "output": str(destination.relative_to(ROOT)),
            "input_sha256": _sha256(path),
        }

    predictions = pd.concat(pieces, ignore_index=True)
    if predictions["text_sha256"].duplicated().any():
        raise RuntimeError("V15 application inputs contain duplicate text hashes")

    audit_rows: list[pd.DataFrame] = []
    audit_summary: dict[str, object] = {}
    for version in range(1, 7):
        audit = _load_audit(version, predictions)
        v14, _ = apply_ambiguity_abstention(
            audit.to_dict(orient="records"), audit["context_prediction"]
        )
        v15, reasons = apply_contextual_ownership_gate_v15(
            audit.to_dict(orient="records"), audit["context_prediction"]
        )
        audit["audit_version"] = version
        audit["v14_prediction"] = v14
        audit["v15_prediction"] = v15
        audit["v15_abstention_reasons"] = reasons
        audit_rows.append(audit)
        audit_summary[f"v{version}"] = {
            "v14": _metrics(audit, pd.Series(v14)),
            "v15": _metrics(audit, pd.Series(v15)),
        }

    pooled = pd.concat(audit_rows, ignore_index=True).drop_duplicates("text_sha256")
    pooled.to_csv(OUTPUT_DIR / "retrospective_audit_predictions.csv", index=False, encoding="utf-8-sig")
    pooled_summary = {
        "v14": _metrics(pooled, pooled["v14_prediction"]),
        "v15": _metrics(pooled, pooled["v15_prediction"]),
    }
    state = {
        "status": "shadow_engineering_complete",
        "model_version": "stance_context_ownership_v15",
        "base_model": "stance_ko_nli_context_v10",
        "active_forecast_changed": False,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "selection_note": (
            "The V15 ownership rule was developed after inspecting prior audits; "
            "all v1-v6 audit metrics are retrospective engineering diagnostics, "
            "not an independent adoption audit."
        ),
        "application": application_summary,
        "audit_by_version": audit_summary,
        "pooled_reused_audits": pooled_summary,
        "artifact_sha256": _sha256(ARTIFACT),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
