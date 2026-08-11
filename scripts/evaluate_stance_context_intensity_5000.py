"""Evaluate continuous stance strength and neutral information on 5,000 rows.

All variants use the same frozen context classifier, the same sentence sample,
and a fixed 0.60 shadow scale.  Neutral information can amplify only an
existing same-issue directional signal; it cannot determine its sign.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402
from scripts import evaluate_neutral_context_protocols as protocol  # noqa: E402
from scripts import evaluate_stance_context_5000_protocols as prior_context  # noqa: E402
from scripts import evaluate_stance_pilot_3000_sensitivity as sensitivity  # noqa: E402
from scripts import evaluate_raw_stance_shadow as shadow  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
INPUT = DATA_DIR / "context_intensity_predictions_5000.csv"
OUTPUT_DIR = DATA_DIR / "intensity_forecast_protocols"
PILOT_DIR = OUTPUT_DIR / "pilot_inputs"
ELECTIONS = protocol.ELECTIONS
PARTY_WEIGHT = 0.65
SPEAKER_ISSUE_CAP = 1.50
ISSUE_CAP = 3.00
MIN_DIRECTIONAL_STRENGTH = 0.08

VARIANTS = (
    {
        "name": "posterior_all",
        "argmax_gate": False,
        "target_information_gain": 0.0,
        "global_information_gain": 0.0,
    },
    {
        "name": "posterior_argmax_gate",
        "argmax_gate": True,
        "target_information_gain": 0.0,
        "global_information_gain": 0.0,
    },
    {
        "name": "posterior_argmax_gate_neutral_info",
        "argmax_gate": True,
        "target_information_gain": 0.25,
        "global_information_gain": 0.15,
    },
)


def _bounded_log(value: float, cap: float) -> float:
    if value <= 0.0:
        return 0.0
    return float(min(np.log1p(value) / np.log1p(cap), 1.0))


def _target_maps(candidates: pd.DataFrame) -> dict[str, dict[tuple[str, str], str]]:
    return {
        "person": {
            (str(row.election_id), str(row.candidate_name)): str(row.slot)
            for row in candidates.itertuples(index=False)
            if str(row.candidate_name).strip()
        },
        "party": {
            (str(row.election_id), str(row.party_name)): str(row.slot)
            for row in candidates.itertuples(index=False)
            if str(row.party_name).strip()
        },
    }


def build_continuous_signals(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    candidates = shadow.candidate_reference()
    candidates = candidates.loc[candidates["election_id"].isin(ELECTIONS)].copy()
    target_maps = _target_maps(candidates)
    candidate_blocs = {
        (str(row.election_id), str(row.slot)): normalize_bloc(row.party_name)
        for row in candidates.itertuples(index=False)
    }
    speaker_blocs = sensitivity._speaker_bloc_lookup()

    global_information: Counter[tuple[str, str]] = Counter()
    target_information: Counter[tuple[str, str, str, str]] = Counter()
    global_seen: set[tuple[str, str, str]] = set()
    target_seen: set[tuple[str, str, str, str, str]] = set()
    for row in frame.itertuples(index=False):
        if str(row.context_model_label) != "neutral":
            continue
        election_id = str(row.election_id)
        issue_name = str(row.issue_name)
        text_hash = str(row.text_sha256)
        information_score = float(row.neutral_information_score)
        if issue_name and text_hash and (election_id, issue_name, text_hash) not in global_seen:
            global_seen.add((election_id, issue_name, text_hash))
            global_information[(election_id, issue_name)] += information_score
        target_type = str(row.target_type)
        target_name = str(row.target_name)
        slot = target_maps.get(target_type, {}).get((election_id, target_name))
        key = (election_id, slot or "", target_type, issue_name, text_hash)
        if slot and issue_name and text_hash and key not in target_seen:
            target_seen.add(key)
            target_information[(election_id, slot, target_type, issue_name)] += information_score

    speaker_issue: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, str, str]] = set()
    argmax_gate = bool(config["argmax_gate"])
    for row in frame.itertuples(index=False):
        target_type = str(row.target_type)
        if target_type not in target_maps:
            continue
        election_id = str(row.election_id)
        target_name = str(row.target_name)
        slot = target_maps[target_type].get((election_id, target_name))
        if not slot:
            continue
        text_hash = str(row.text_sha256)
        dedup_key = (election_id, text_hash, target_type, target_name)
        if not text_hash or dedup_key in seen:
            continue
        seen.add(dedup_key)
        if argmax_gate and str(row.context_model_label) == "neutral":
            continue
        directional_strength = float(row.directional_strength)
        if directional_strength < MIN_DIRECTIONAL_STRENGTH:
            continue
        directional_score = float(row.directional_score)
        issue_name = str(row.issue_name) or "unknown_issue"
        issue_weight = float(row.issue_weight or 0.0)
        speaker = str(row.speaker) or f"unknown::{row.source_row_id}"
        speaker_key = sensitivity.normalize_speaker_name(speaker)
        speaker_bloc = speaker_blocs.get((str(row.assembly_daesu), speaker_key), "")
        target_bloc = candidate_blocs.get((election_id, slot), "")
        context_multiplier = 0.70
        if speaker_bloc and target_bloc:
            if speaker_bloc == target_bloc:
                context_multiplier = 1.20 if directional_score < 0.0 else 1.00
            else:
                context_multiplier = 0.75 if directional_score < 0.0 else 1.20
        weight = issue_weight * context_multiplier
        cell = speaker_issue[(election_id, slot, target_type, issue_name, speaker)]
        cell["signed"] += directional_score * weight
        cell["absolute"] += directional_strength * weight
        cell["evidence_count"] += 1

    issues: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for (election_id, slot, target_type, issue_name, _speaker), cell in speaker_issue.items():
        issue = issues[(election_id, slot, target_type, issue_name)]
        issue["signed"] += float(np.clip(cell["signed"], -SPEAKER_ISSUE_CAP, SPEAKER_ISSUE_CAP))
        issue["absolute"] += min(float(cell["absolute"]), SPEAKER_ISSUE_CAP)
        issue["evidence_count"] += cell["evidence_count"]
        issue["speaker_count"] += 1

    totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    target_gain = float(config["target_information_gain"])
    global_gain = float(config["global_information_gain"])
    for (election_id, slot, target_type, issue_name), issue in issues.items():
        target_info = _bounded_log(
            target_information[(election_id, slot, target_type, issue_name)], 20.0
        )
        global_info = _bounded_log(global_information[(election_id, issue_name)], 100.0)
        information_multiplier = 1.0 + target_gain * target_info + global_gain * global_info
        signed = float(issue["signed"]) * information_multiplier
        absolute = float(issue["absolute"]) * information_multiplier
        cap_factor = min(1.0, ISSUE_CAP / max(absolute, 1e-12))
        total = totals[(election_id, slot, target_type)]
        total["signed"] += signed * cap_factor
        total["absolute"] += absolute * cap_factor
        total["evidence_count"] += issue["evidence_count"]
        total["issue_count"] += 1
        total["speaker_count"] += issue["speaker_count"]
        total["target_information_sum"] += target_info
        total["global_information_sum"] += global_info

    out = candidates.copy()
    for target_type in ("person", "party"):
        keys = [
            (str(row.election_id), str(row.slot), target_type)
            for row in out.itertuples(index=False)
        ]
        for column in (
            "signed",
            "absolute",
            "evidence_count",
            "issue_count",
            "speaker_count",
            "target_information_sum",
            "global_information_sum",
        ):
            out[f"{target_type}_{column}"] = [totals[key][column] for key in keys]
        net = out[f"{target_type}_signed"] / (out[f"{target_type}_absolute"] + 4.0)
        coverage = np.minimum(np.sqrt(out[f"{target_type}_evidence_count"]) / 8.0, 1.0)
        issue_breadth = np.minimum(np.sqrt(out[f"{target_type}_issue_count"]) / 3.0, 1.0)
        out[f"{target_type}_signal"] = net * coverage * (0.75 + 0.25 * issue_breadth)

    out["evidence_count"] = out["person_evidence_count"] + out["party_evidence_count"]
    out["covered_slot_count"] = out["evidence_count"].gt(0).groupby(out["election_id"]).transform("sum")
    out["coverage_gate_passed"] = out["covered_slot_count"].ge(2).astype(int)
    out["stance_shadow_signal"] = (
        out["person_signal"] + PARTY_WEIGHT * out["party_signal"]
    ) * out["coverage_gate_passed"]
    out["stance_shadow_signal"] -= out.groupby("election_id")["stance_shadow_signal"].transform("mean")
    out.insert(0, "variant", str(config["name"]))
    return out


def materialize_existing_aggregator_variants(frame: pd.DataFrame) -> None:
    """Keep the prior aggregator fixed and alter only sentence-level weights."""
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    for election_id in ELECTIONS:
        election = frame.loc[frame["election_id"].eq(election_id)].copy()
        if len(election) != 1_000:
            raise RuntimeError(f"{election_id}: expected 1,000 rows, found {len(election)}")
        for variant, filter_information in (
            ("context_strength_weighted", False),
            ("context_strength_weighted_info_filter", True),
        ):
            pilot = election.copy()
            pilot["rule_stance_polarity"] = pilot["context_model_polarity"].astype(int)
            pilot["rule_stance_label"] = pilot["context_model_label"].map(
                {"negative": "attack", "neutral": "neutral", "positive": "endorse"}
            )
            pilot["rule_stance_confidence"] = pilot["context_model_probability"].astype(float)
            directional = ~pilot["context_model_label"].eq("neutral")
            ratio = np.divide(
                pilot["directional_strength"].to_numpy(float),
                pilot["context_model_probability"].to_numpy(float),
                out=np.zeros(len(pilot), dtype=float),
                where=pilot["context_model_probability"].to_numpy(float) > 0.0,
            )
            ratio = np.clip(ratio, 0.0, 1.0)
            original_issue_weight = pd.to_numeric(
                pilot["issue_weight"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            pilot["issue_weight"] = np.where(
                directional, original_issue_weight * ratio, original_issue_weight
            )
            if filter_information:
                uninformative = pilot["context_model_label"].eq("neutral") & pilot[
                    "neutral_information_label"
                ].isin(["none", "low"])
                pilot.loc[uninformative, "rule_stance_label"] = "noninformative"
            pilot.to_csv(
                PILOT_DIR / f"{variant}_{election_id}.csv",
                index=False,
                encoding="utf-8-sig",
            )


def existing_aggregator_signals(variant: str) -> pd.DataFrame:
    config = next(
        config for config in sensitivity.CONFIGS if config["name"] == protocol.CONFIG_NAME
    )
    pieces: list[pd.DataFrame] = []
    for election_id in ELECTIONS:
        features = sensitivity.build_features(
            config, pilot_input=PILOT_DIR / f"{variant}_{election_id}.csv"
        )
        piece = features.loc[
            features["election_id"].eq(election_id),
            ["election_id", "slot", "stance_shadow_signal"],
        ].copy()
        piece.insert(0, "variant", variant)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def evaluate_variant(frame: pd.DataFrame, signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    selected = signals.loc[
        signals["variant"].eq(variant), ["election_id", "slot", "stance_shadow_signal"]
    ]
    rows = pd.concat(
        [
            protocol.full_fit_rows(frame, selected),
            protocol.loeo_rows(frame, selected),
            protocol.rolling_rows(frame, selected),
        ],
        ignore_index=True,
    )
    rows.insert(0, "variant", variant)
    return rows


def main() -> None:
    predictions = pd.read_csv(INPUT, encoding="utf-8-sig").fillna("")
    if len(predictions) != 5_000 or predictions["text_sha256"].nunique() != 5_000:
        raise RuntimeError("intensity input must contain 5,000 unique sentence hashes")

    prior_path = DATA_DIR / "forecast_protocols" / "candidate_signals.csv"
    if not prior_path.exists():
        prior_context.materialize_pilots()
        prior_signals = pd.concat(
            [prior_context.signals_for("legacy"), prior_context.signals_for("context")],
            ignore_index=True,
        )
    else:
        prior_signals = pd.read_csv(prior_path, encoding="utf-8-sig")
    prior_signals["variant"] = prior_signals["variant"].replace(
        {"legacy": "legacy_rule", "context": "context_argmax"}
    )

    materialize_existing_aggregator_variants(predictions)
    fixed_aggregator_signals = pd.concat(
        [
            existing_aggregator_signals("context_strength_weighted"),
            existing_aggregator_signals("context_strength_weighted_info_filter"),
        ],
        ignore_index=True,
    )

    continuous = pd.concat(
        [build_continuous_signals(predictions, config) for config in VARIANTS],
        ignore_index=True,
    )
    diagnostic_columns = [
        column
        for column in continuous.columns
        if column
        in {
            "variant",
            "election_id",
            "slot",
            "candidate_name",
            "party_name",
            "stance_shadow_signal",
            "evidence_count",
            "covered_slot_count",
            "coverage_gate_passed",
        }
        or column.startswith("person_")
        or column.startswith("party_")
    ]
    signals = pd.concat(
        [
            prior_signals[["variant", "election_id", "slot", "stance_shadow_signal"]],
            fixed_aggregator_signals[
                ["variant", "election_id", "slot", "stance_shadow_signal"]
            ],
            continuous[["variant", "election_id", "slot", "stance_shadow_signal"]],
        ],
        ignore_index=True,
    )
    variants = signals["variant"].drop_duplicates().tolist()
    assembled = protocol.engine.assemble()
    frame = assembled.loc[assembled["election_id"].isin(ELECTIONS)].copy()
    rows = pd.concat(
        [evaluate_variant(frame, signals, variant) for variant in variants], ignore_index=True
    )

    row_summaries: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    national_summaries: list[pd.DataFrame] = []
    for variant, group in rows.groupby("variant", sort=False):
        row_summary = protocol.row_summary(group.drop(columns="variant"))
        row_summary.insert(0, "variant", variant)
        row_summaries.append(row_summary)
        points = protocol.national_points(group.drop(columns="variant"))
        points.insert(0, "variant", variant)
        point_frames.append(points)
        national_summary = protocol.national_summary(points.drop(columns="variant"))
        national_summary.insert(0, "variant", variant)
        national_summaries.append(national_summary)

    row_summary = pd.concat(row_summaries, ignore_index=True)
    points = pd.concat(point_frames, ignore_index=True)
    national_summary = pd.concat(national_summaries, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "sentence_labels_5000.csv", index=False, encoding="utf-8-sig")
    continuous[diagnostic_columns].to_csv(
        OUTPUT_DIR / "continuous_candidate_signal_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    signals.to_csv(OUTPUT_DIR / "candidate_signals.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(OUTPUT_DIR / "protocol_row_predictions.csv", index=False, encoding="utf-8-sig")
    row_summary.to_csv(OUTPUT_DIR / "protocol_row_summary.csv", index=False, encoding="utf-8-sig")
    points.to_csv(OUTPUT_DIR / "protocol_national_points.csv", index=False, encoding="utf-8-sig")
    national_summary.to_csv(
        OUTPUT_DIR / "protocol_national_summary.csv", index=False, encoding="utf-8-sig"
    )

    print(row_summary.loc[row_summary["election_id"].eq("Overall")].to_string(index=False))
    print()
    print(national_summary.loc[national_summary["election_id"].eq("Overall")].to_string(index=False))


if __name__ == "__main__":
    main()
