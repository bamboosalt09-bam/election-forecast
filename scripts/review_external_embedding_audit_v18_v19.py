"""Adjudicate the frozen V18-V19 V29-S audit without editing lock files."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    precision_first_metrics,
    stance_adoption_assessment,
)


OVERRIDES = {
    "stance_v18_004": (
        "neutral",
        True,
        "reported_external",
        "quoted economic-crisis claim is described as an unusual remark, not adopted",
    ),
    "stance_v18_005": (
        "neutral",
        False,
        "fragment",
        "party is only present in an incomplete office-history list",
    ),
    "stance_v18_006": (
        "neutral",
        False,
        "reported_external",
        "reports criticism of Korean and Thai officials and asks for the prime minister's view",
    ),
    "stance_v18_013": (
        "neutral",
        True,
        "reported_opposition_frame",
        "describes the opposition party's frame without adopting it",
    ),
    "stance_v18_015": (
        "neutral",
        False,
        "historical_other_government",
        "criticism concerns the prior Lee Myung-bak government in the 2017 window",
    ),
    "stance_v18_023": (
        "neutral",
        True,
        "reported_market_view",
        "quotes a market saying about government policy",
    ),
    "stance_v19_001": (
        "neutral",
        True,
        "quotation_continuation",
        "current sentence continues the external quotation opened in prior context",
    ),
    "stance_v19_012": (
        "neutral",
        True,
        "reported_external",
        "reports another person's evaluation and asks for a response",
    ),
    "stance_v19_041": (
        "positive",
        True,
        "speaker",
        "credits the party with anticipating market stress; criticism is directed at government",
    ),
    "stance_v19_053": (
        "neutral",
        False,
        "historical_other_government",
        "only the prior Lee Myung-bak government is criticized in the 2017 window",
    ),
    "stance_v19_054": (
        "neutral",
        False,
        "nominee_is_target",
        "criticism concerns the prime-minister nominee and his philosophy, not government",
    ),
    "stance_v19_061": (
        "positive",
        True,
        "speaker",
        "defends the Lee government as inheriting the housing downturn rather than causing it",
    ),
    "stance_v19_064": (
        "neutral",
        True,
        "reported_public_view",
        "attributes dissatisfaction to the public and asks whether the prime minister knows it",
    ),
    "stance_v19_065": (
        "neutral",
        True,
        "policy_prescription",
        "prescribes direct government action while criticizing firms and markets",
    ),
    "stance_v19_075": (
        "neutral",
        True,
        "quotation_unknown",
        "current sentence is an unattributed quotation",
    ),
    "stance_v19_076": (
        "neutral",
        False,
        "target_self_report",
        "Moon is the speaker of a historical quotation criticizing government, not its object",
    ),
    "stance_v19_081": (
        "neutral",
        False,
        "other_actor_is_target",
        "criticizes irresponsible pessimistic commentators rather than government",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    frames: list[pd.DataFrame] = []
    lock_hashes: dict[str, str] = {}
    for version in (18, 19):
        lock_path = ROOT / "data" / "shadow" / f"stance_locked_audit_v{version}.csv"
        labels_path = ROOT / "data" / "shadow" / f"stance_locked_audit_v{version}_labels.csv"
        if labels_path.exists():
            raise FileExistsError(labels_path)
        frame = pd.read_csv(lock_path, encoding="utf-8-sig").fillna("")
        if any(frame[column].astype(str).str.len().gt(0).any() for column in (
            "audit_locked_label",
            "audit_target_correct",
            "audit_quotation_owner",
            "audit_notes",
        )):
            raise ValueError(f"lock file already contains adjudication: {lock_path}")
        rows: list[dict[str, object]] = []
        for row in frame.to_dict(orient="records"):
            audit_id = str(row["audit_id"])
            label, target_correct, owner, notes = OVERRIDES.get(
                audit_id,
                (
                    str(row["v29_prediction"]),
                    True,
                    "speaker",
                    "direct speaker-owned evaluation of the assigned target",
                ),
            )
            rows.append(
                {
                    "text_sha256": row["text_sha256"],
                    "audit_locked_label": label,
                    "audit_target_correct": target_correct,
                    "audit_quotation_owner": owner,
                    "audit_notes": notes,
                }
            )
        labels = pd.DataFrame(rows)
        labels.to_csv(labels_path, index=False, encoding="utf-8-sig")
        reviewed = frame.drop(
            columns=[
                "audit_locked_label",
                "audit_target_correct",
                "audit_quotation_owner",
                "audit_notes",
            ]
        ).merge(labels, on="text_sha256", validate="one_to_one")
        reviewed["audit_version"] = version
        reviewed["review_label"] = reviewed["audit_locked_label"].where(
            reviewed["audit_target_correct"].astype(bool), "neutral"
        )
        frames.append(reviewed)
        lock_hashes[f"v{version}"] = sha256_file(lock_path)

    combined = pd.concat(frames, ignore_index=True)
    metrics = precision_first_metrics(
        combined["review_label"], combined["v29_prediction"]
    )
    payload = {
        "status": "independent_audit_complete_not_promoted",
        "model_version": "stance_external_embedding_role_veto_v29s",
        "lock_hashes": lock_hashes,
        "rows": len(combined),
        "label_distribution": combined["review_label"].value_counts().to_dict(),
        "metrics": metrics,
        "by_lock": {
            f"v{version}": precision_first_metrics(
                combined.loc[combined["audit_version"].eq(version), "review_label"],
                combined.loc[combined["audit_version"].eq(version), "v29_prediction"],
            )
            for version in (18, 19)
        },
        "adoption": {
            **stance_adoption_assessment(
                metrics,
                independent_audit=True,
                target_attribution_audited=True,
                point_in_time_audited=True,
                rolling_non_degradation=False,
            ),
            "active_forecast_changed": False,
        },
        "vote_outcomes_used": False,
        "post_2022_rows_present": False,
    }
    output = (
        ROOT
        / "outputs"
        / "assembly_stance"
        / "stance_external_embedding_role_veto_v29s"
        / "locked_audit_v18_v19_metrics.json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
