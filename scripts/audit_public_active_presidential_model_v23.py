"""Run the frozen V23 audit when private bulk inputs are absent from a clone."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_active_presidential_model_v23 as audit


EXTERNAL_INPUTS = (
    audit.ROOT / "data" / "raw" / "official_sources" / "external_active_inputs.json"
)


def _audit_active_manifest_with_external_records() -> dict[str, int]:
    manifest = pd.read_csv(audit.ACTIVE_DIR / "input_manifest.csv", encoding="utf-8-sig")
    payload = json.loads(EXTERNAL_INPUTS.read_text(encoding="utf-8"))
    external_inputs = {
        str(row["path"]).replace("\\", "/"): row for row in payload.get("inputs", [])
    }
    paths = manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    audit._require(
        not (set(paths) & audit.LEGACY_TRANSFER_INPUTS),
        "legacy transfer input is active",
    )
    required = {
        "outputs/automatic_controls_v23/withdrawal_transfer_registry.csv",
        "outputs/automatic_controls_v23/election_generation_weights.csv",
        "outputs/automatic_controls_v23/candidate_political_profiles.csv",
        "data/raw/withdrawal_events.csv",
        "data/raw/official_sources/assembly_candidate_attention_history.csv",
        "data/raw/official_sources/nec_age_turnout_composition_history.csv",
    }
    audit._require(required.issubset(set(paths)), "V23 manifest is missing required inputs")

    for row in manifest.itertuples(index=False):
        path_text = str(row.path).replace("\\", "/")
        if path_text.startswith("generated:"):
            continue
        path = audit.ROOT / path_text
        if path.exists():
            audit._require(
                audit._sha256(path) == str(row.sha256),
                f"manifest hash drift: {path_text}",
            )
            continue

        external = external_inputs.get(path_text)
        audit._require(external is not None, f"manifest input is missing: {path_text}")
        audit._require(
            external.get("status") == "excluded_from_git",
            f"external input has an invalid status: {path_text}",
        )
        audit._require(
            int(external.get("bytes", -1)) == int(row.bytes)
            and str(external.get("sha256", "")) == str(row.sha256),
            f"external input record drift: {path_text}",
        )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path_text],
            cwd=audit.ROOT,
            capture_output=True,
        )
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", path_text],
            cwd=audit.ROOT,
            capture_output=True,
        )
        audit._require(tracked.returncode != 0, f"external input is tracked: {path_text}")
        audit._require(ignored.returncode == 0, f"external input is not ignored: {path_text}")
    return {"active_manifest_files": len(manifest)}


def main() -> None:
    audit._audit_active_manifest = _audit_active_manifest_with_external_records
    audit.main()


if __name__ == "__main__":
    main()
