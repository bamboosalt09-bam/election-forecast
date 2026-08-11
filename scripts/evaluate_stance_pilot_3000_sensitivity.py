"""Evaluate scale and confidence sensitivity on the 2022 3,000-row review batch.

This remains a non-PIT diagnostic. The batch is stratified and its rule labels
are not human-validated, so the best 2022 setting must not be promoted into the
active forecast engine.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import evaluate_raw_stance_shadow as shadow  # noqa: E402
from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402


PILOT_INPUT = ROOT / "outputs" / "assembly_stance" / "pilot_pres_2022_3000" / "review_batch.csv"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "pilot_pres_2022_3000" / "sensitivity"
MEMBER_HISTORY = ROOT / "data" / "raw" / "assembly_member_history.csv"
ASSEMBLY_ROSTER = ROOT / "data" / "assembly_roster.csv"
ASSEMBLY15_ROSTER = ROOT / "data" / "raw" / "assembly15_member_roster.csv"
SCALES = (
    0.006, 0.030, 0.060, 0.090, 0.120, 0.180, 0.240, 0.300,
    0.360, 0.420, 0.480, 0.540, 0.600,
)
CONFIGS = (
    {"name": "direct_conf_060", "party_weight": 0.00, "attention_gain": 0.00, "min_confidence": 0.60, "min_covered_slots": 0},
    {"name": "person_party_directional", "party_weight": 0.65, "attention_gain": 0.00, "min_confidence": 0.60, "min_covered_slots": 0},
    {"name": "person_party_directional_gate2", "party_weight": 0.65, "attention_gain": 0.00, "min_confidence": 0.60, "min_covered_slots": 2},
    {
        "name": "person_party_speaker_context_gate2",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "min_confidence": 0.60,
        "min_covered_slots": 2,
        "speaker_context": True,
    },
    {
        "name": "person_party_speaker_confirmed_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "min_confidence": 0.60,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf2_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "min_confidence": 0.60,
        "confidence_power": 2.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_min065_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "min_confidence": 0.65,
        "confidence_power": 1.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_context020_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.20,
        "min_confidence": 0.60,
        "confidence_power": 1.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_context035_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.35,
        "min_confidence": 0.60,
        "confidence_power": 1.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf2_context020_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.20,
        "min_confidence": 0.60,
        "confidence_power": 2.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context020_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.20,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf2_context035_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.35,
        "min_confidence": 0.60,
        "confidence_power": 2.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context035_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.35,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_context050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "min_confidence": 0.60,
        "confidence_power": 1.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_context075_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.75,
        "min_confidence": 0.60,
        "confidence_power": 1.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context075_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.75,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context100_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 1.00,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context125_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 1.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context150_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 1.50,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_global025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 0.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_global050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 0.50,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_global075_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 0.75,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_global100_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 1.00,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_global025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_gain": 0.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_global050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_gain": 0.50,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_global075_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_gain": 0.75,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_globalrel025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 0.25,
        "neutral_global_metric": "relative_strength",
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_globalrel050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 0.50,
        "neutral_global_metric": "relative_strength",
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_globalrel075_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_gain": 0.75,
        "neutral_global_metric": "relative_strength",
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_globalrel025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_gain": 0.25,
        "neutral_global_metric": "relative_strength",
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_globalrel050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_gain": 0.50,
        "neutral_global_metric": "relative_strength",
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_issueglobal025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_issue_gain": 0.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_issueglobal050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_issue_gain": 0.50,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_issueglobal075_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_global_issue_gain": 0.75,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_issueglobal025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_issue_gain": 0.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context050_issueglobal050_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.50,
        "neutral_global_issue_gain": 0.50,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context075_issueglobal025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 0.75,
        "neutral_global_issue_gain": 0.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {
        "name": "person_party_speaker_confirmed_conf3_context100_issueglobal025_gate2",
        "aggregation": "speaker_confirmation",
        "party_weight": 0.65,
        "attention_gain": 0.00,
        "neutral_context_gain": 1.00,
        "neutral_global_issue_gain": 0.25,
        "min_confidence": 0.60,
        "confidence_power": 3.0,
        "min_covered_slots": 2,
    },
    {"name": "person_party_attention_035", "party_weight": 0.65, "attention_gain": 0.35, "min_confidence": 0.60, "min_covered_slots": 0},
    {"name": "person_party_attention_070", "party_weight": 0.65, "attention_gain": 0.70, "min_confidence": 0.60, "min_covered_slots": 0},
    {
        "name": "rigorous_speaker_issue",
        "aggregation": "speaker_issue",
        "party_weight": 0.65,
        "conflict_party_weight": 0.65,
        "min_confidence": 0.60,
        "min_covered_slots": 2,
    },
    {
        "name": "rigorous_speaker_issue_conflict",
        "aggregation": "speaker_issue",
        "party_weight": 0.65,
        "conflict_party_weight": 0.25,
        "min_confidence": 0.60,
        "min_covered_slots": 2,
    },
)

LABEL_RELIABILITY = {
    "attack": 1.00,
    "defend": 0.85,
    "endorse": 0.80,
    "rebuttal": 0.50,
}
SPEAKER_ISSUE_CAP = 1.50
ISSUE_CAP = 3.00
NEUTRAL_ANALYSIS_RE = re.compile(
    r"원인|영향|결과|전망|현황|대책|방안|대안|필요|때문|따라서|분석|평가|통계|자료|조사"
)
NEUTRAL_IMPACT_RE = re.compile(
    r"문제|위기|심각|우려|의혹|논란|실패|악화|부족|침체|부담|불안|피해|위법|부패|"
    r"폭등|폭락|개선|회복|증가|감소|성장|강화|확대|축소|성과|안정|해결|정상화"
)


def neutral_content_flags(value: object) -> tuple[int, int]:
    """Return analysis and impact flags without assigning vote direction."""
    text = str(value or "")
    return int(bool(NEUTRAL_ANALYSIS_RE.search(text))), int(bool(NEUTRAL_IMPACT_RE.search(text)))


def meeting_quarter(value: object) -> str:
    match = re.match(r"^(\d{4})-(\d{2})", str(value or ""))
    if not match:
        return ""
    return f"{match.group(1)}-Q{(int(match.group(2)) - 1) // 3 + 1}"


def bounded_log_strength(value: float, cap: float) -> float:
    if value <= 0.0 or cap <= 0.0:
        return 0.0
    return float(min(np.log1p(value) / np.log1p(cap), 1.0))


def normalize_speaker_name(value: object) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(value or ""))
    text = re.sub(
        r"(위원장|부위원장|위원|의원|장관|차관|국무총리|총리|의장|부의장|대표|대리)",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    names = re.findall(r"(?<![가-힣])([가-힣]{2,4})(?![가-힣])", text)
    return names[-1] if names else text


def _speaker_bloc_lookup() -> dict[tuple[str, str], str]:
    pieces = [
        pd.read_csv(ASSEMBLY_ROSTER, dtype=str)[["daesu", "name", "bloc"]],
        pd.read_csv(ASSEMBLY15_ROSTER, dtype=str)[["daesu", "name", "bloc"]],
        pd.read_csv(MEMBER_HISTORY, dtype=str)[["daesu", "name", "bloc"]],
    ]
    history = pd.concat(pieces, ignore_index=True).fillna("")
    history["speaker_key"] = history["name"].map(normalize_speaker_name)
    history["normalized_bloc"] = history["bloc"].map(normalize_bloc)
    lookup: dict[tuple[str, str], str] = {}
    for (daesu, speaker_key), group in history.groupby(["daesu", "speaker_key"], sort=False):
        blocs = [value for value in group["normalized_bloc"].astype(str) if value]
        if not speaker_key or not blocs:
            continue
        counts = Counter(blocs)
        most_common = counts.most_common(2)
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            lookup[(str(daesu), str(speaker_key))] = most_common[0][0]
    return lookup


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


def build_rigorous_features(config: dict[str, object], pilot_input: Path) -> pd.DataFrame:
    candidates = shadow.candidate_reference()
    target_maps = _target_maps(candidates)
    candidate_blocs = {
        (str(row.election_id), str(row.slot)): normalize_bloc(row.party_name)
        for row in candidates.itertuples(index=False)
    }
    speaker_blocs = _speaker_bloc_lookup()
    min_confidence = float(config["min_confidence"])
    min_covered_slots = int(config.get("min_covered_slots", 0))
    party_weight = float(config["party_weight"])
    conflict_party_weight = float(config.get("conflict_party_weight", party_weight))
    groups: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    attention: Counter[tuple[str, str, str]] = Counter()
    seen: set[tuple[str, str, str, str]] = set()

    with pilot_input.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target_type = row.get("target_type", "")
            if target_type not in target_maps:
                continue
            election_id = row.get("election_id", "")
            target = row.get("target_name", "")
            slot = target_maps[target_type].get((election_id, target))
            if not slot:
                continue
            text_hash = row.get("text_sha256", "")
            dedup_key = (election_id, text_hash, target_type, target)
            if not text_hash or dedup_key in seen:
                continue
            seen.add(dedup_key)
            attention[(election_id, slot, target_type)] += 1
            label = row.get("rule_stance_label", "")
            if label not in LABEL_RELIABILITY:
                continue
            confidence = float(row.get("rule_stance_confidence") or 0.0)
            if confidence < min_confidence:
                continue
            issue_name = row.get("issue_name", "") or "unknown_issue"
            speaker = row.get("speaker", "") or f"unknown::{row.get('source_row_id', '')}"
            issue_weight = float(row.get("issue_weight") or 0.0)
            polarity = float(row.get("rule_stance_polarity") or 0.0)
            weight = confidence * np.sqrt(np.clip(issue_weight, 0.25, 2.25)) * LABEL_RELIABILITY[label]
            cell = groups[(election_id, slot, target_type, issue_name, speaker)]
            cell["signed"] += polarity * weight
            cell["absolute"] += abs(polarity) * weight
            cell["row_count"] += 1

    issues: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    issue_speakers: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for (election_id, slot, target_type, issue_name, speaker), cell in groups.items():
        issue_key = (election_id, slot, target_type, issue_name)
        issues[issue_key]["signed"] += float(np.clip(cell["signed"], -SPEAKER_ISSUE_CAP, SPEAKER_ISSUE_CAP))
        issues[issue_key]["absolute"] += min(float(cell["absolute"]), SPEAKER_ISSUE_CAP)
        issues[issue_key]["row_count"] += cell["row_count"]
        issue_speakers[issue_key].add(speaker)

    totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    total_speakers: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for issue_key, cell in issues.items():
        election_id, slot, target_type, _ = issue_key
        total_key = (election_id, slot, target_type)
        cap_factor = min(1.0, ISSUE_CAP / max(float(cell["absolute"]), 1e-12))
        totals[total_key]["signed"] += float(cell["signed"]) * cap_factor
        totals[total_key]["absolute"] += float(cell["absolute"]) * cap_factor
        totals[total_key]["evidence_count"] += cell["row_count"]
        totals[total_key]["issue_count"] += 1
        total_speakers[total_key].update(issue_speakers[issue_key])

    out = candidates.copy()
    for target_type in ("person", "party"):
        keys = [(str(row.election_id), str(row.slot), target_type) for row in out.itertuples(index=False)]
        for column in ("signed", "absolute", "evidence_count", "issue_count"):
            out[f"{target_type}_{column}"] = [totals[key][column] for key in keys]
        out[f"{target_type}_attention_count"] = [attention[key] for key in keys]
        out[f"{target_type}_speaker_count"] = [len(total_speakers[key]) for key in keys]
        net = out[f"{target_type}_signed"] / (out[f"{target_type}_absolute"] + 4.0)
        speaker_coverage = np.minimum(np.sqrt(out[f"{target_type}_speaker_count"]) / 4.0, 1.0)
        issue_breadth = np.minimum(np.sqrt(out[f"{target_type}_issue_count"]) / 3.0, 1.0)
        out[f"{target_type}_signal"] = net * speaker_coverage * (0.75 + 0.25 * issue_breadth)

    out["evidence_count"] = out["person_evidence_count"] + out["party_evidence_count"]
    out["attention_count"] = out["person_attention_count"] + out["party_attention_count"]
    out["context_neutral_count"] = 0
    out["context_issue_overlap_count"] = 0
    out["speaker_context_mapped_count"] = 0
    out["same_bloc_count"] = 0
    out["other_bloc_count"] = 0
    out["covered_slot_count"] = out["evidence_count"].gt(0).groupby(out["election_id"]).transform("sum")
    out["coverage_gate_passed"] = out["covered_slot_count"].ge(min_covered_slots).astype(int)
    conflict = out["person_signal"].mul(out["party_signal"]).lt(0.0)
    effective_party_weight = np.where(conflict, conflict_party_weight, party_weight)
    out["stance_net"] = out["person_signal"] + effective_party_weight * out["party_signal"]
    out["stance_net"] *= out["coverage_gate_passed"]
    out["stance_shadow_signal"] = out["stance_net"]
    out["stance_shadow_signal"] -= out.groupby("election_id")["stance_shadow_signal"].transform("mean")
    return out


def build_features(
    config: dict[str, object],
    *,
    pilot_input: Path = PILOT_INPUT,
    progress_every: int = 0,
) -> pd.DataFrame:
    if config.get("aggregation") == "speaker_confirmation":
        base_config = {
            "party_weight": config["party_weight"],
            "attention_gain": 0.0,
            "min_confidence": config["min_confidence"],
            "min_covered_slots": config["min_covered_slots"],
            "confidence_power": config.get("confidence_power", 1.0),
            "neutral_global_issue_gain": float(config.get("neutral_global_issue_gain", 0.0)),
            "neutral_global_issue_metric": str(config.get("neutral_global_issue_metric", "strength")),
            "extract_neutral_global": (
                float(config.get("neutral_global_gain", 0.0)) > 0.0
                or float(config.get("neutral_global_issue_gain", 0.0)) > 0.0
            ),
        }
        context_config = {**base_config, "speaker_context": True}
        base = build_features(
            base_config,
            pilot_input=pilot_input,
            progress_every=progress_every,
        )
        context = build_features(
            context_config,
            pilot_input=pilot_input,
            progress_every=progress_every,
        )
        base_signal = base["stance_shadow_signal"].to_numpy(float)
        context_signal = context["stance_shadow_signal"].to_numpy(float)
        evidence = context["evidence_count"].replace(0, np.nan).to_numpy(float)
        mapped_ratio = np.divide(
            context["speaker_context_mapped_count"].to_numpy(float),
            evidence,
            out=np.zeros(len(context), dtype=float),
            where=np.isfinite(evidence),
        )
        agrees = base_signal * context_signal > 0.0
        conflicts = base_signal * context_signal < 0.0
        multiplier = np.where(agrees, 1.0 + 0.15 * mapped_ratio, 0.75)
        multiplier = np.where(conflicts, 0.50, multiplier)
        confirmed_signal = base_signal * multiplier
        confirmed_signal[np.abs(base_signal) < 0.002] = 0.0
        neutral_context_gain = float(config.get("neutral_context_gain", 0.0))
        if neutral_context_gain > 0.0:
            context_volume = np.minimum(
                np.log1p(base["context_neutral_count"].to_numpy(float)) / np.log1p(40.0),
                1.0,
            )
            context_breadth = np.minimum(
                base["context_issue_overlap_count"].to_numpy(float) / 3.0,
                1.0,
            )
            context_strength = 0.7 * context_volume + 0.3 * context_breadth
            confirmed_signal *= 1.0 + neutral_context_gain * context_strength
        neutral_global_gain = float(config.get("neutral_global_gain", 0.0))
        if neutral_global_gain > 0.0:
            neutral_global_metric = str(config.get("neutral_global_metric", "strength"))
            if neutral_global_metric not in {"strength", "relative_strength"}:
                raise ValueError(f"unsupported neutral global metric: {neutral_global_metric}")
            confirmed_signal *= (
                1.0
                + neutral_global_gain
                * base[f"global_context_{neutral_global_metric}"].to_numpy(float)
            )
        base["stance_net"] = confirmed_signal
        base["stance_shadow_signal"] = confirmed_signal
        for column in (
            "speaker_context_mapped_count", "same_bloc_count", "other_bloc_count",
            "person_speaker_context_mapped_count", "party_speaker_context_mapped_count",
            "person_same_bloc_count", "party_same_bloc_count",
            "person_other_bloc_count", "party_other_bloc_count",
        ):
            base[column] = context[column]
        return base
    if config.get("aggregation") == "speaker_issue":
        return build_rigorous_features(config, pilot_input)
    candidates = shadow.candidate_reference()
    target_maps = _target_maps(candidates)
    candidate_blocs = {
        (str(row.election_id), str(row.slot)): normalize_bloc(row.party_name)
        for row in candidates.itertuples(index=False)
    }
    extract_neutral_global = bool(config.get("extract_neutral_global", False))
    all_speaker_blocs = _speaker_bloc_lookup() if config.get("speaker_context") or extract_neutral_global else {}
    speaker_blocs = all_speaker_blocs if config.get("speaker_context") else {}
    min_confidence = float(config["min_confidence"])
    confidence_power = float(config.get("confidence_power", 1.0))
    attention_gain = float(config["attention_gain"])
    party_weight = float(config["party_weight"])
    min_covered_slots = int(config.get("min_covered_slots", 0))
    totals: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, str, str]] = set()
    issue_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    directional_issue_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    neutral_issue_counts: Counter[tuple[str, str, str, str]] = Counter()
    directional_issue_mass: Counter[tuple[str, str, str, str]] = Counter()
    directional_issue_signed: Counter[tuple[str, str, str, str]] = Counter()
    directional_issue_absolute: Counter[tuple[str, str, str, str]] = Counter()
    global_neutral_counts: Counter[tuple[str, str]] = Counter()
    global_neutral_analysis_counts: Counter[tuple[str, str]] = Counter()
    global_neutral_impact_counts: Counter[tuple[str, str]] = Counter()
    global_neutral_speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
    global_neutral_committees: dict[tuple[str, str], set[str]] = defaultdict(set)
    global_neutral_periods: dict[tuple[str, str], set[str]] = defaultdict(set)
    global_neutral_blocs: dict[tuple[str, str], set[str]] = defaultdict(set)
    global_neutral_seen: set[tuple[str, str, str]] = set()

    with pilot_input.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            if progress_every > 0 and row_number % progress_every == 0:
                print(f"[feature scan] {pilot_input.name}: {row_number:,} rows", flush=True)
            target_type = row.get("target_type", "")
            label = row.get("rule_stance_label") or row.get("stance_label", "")
            election_id = row.get("election_id", "")
            issue_name = row.get("issue_name", "")
            text_hash = row.get("text_sha256", "")
            if extract_neutral_global and label in {"", "neutral", "ambiguous"} and issue_name and text_hash:
                global_key = (election_id, issue_name)
                global_seen_key = (election_id, issue_name, text_hash)
                if global_seen_key not in global_neutral_seen:
                    global_neutral_seen.add(global_seen_key)
                    global_neutral_counts[global_key] += 1
                    analysis_flag, impact_flag = neutral_content_flags(row.get("text_excerpt", ""))
                    global_neutral_analysis_counts[global_key] += analysis_flag
                    global_neutral_impact_counts[global_key] += impact_flag
                    speaker_key = normalize_speaker_name(row.get("speaker", ""))
                    if speaker_key:
                        global_neutral_speakers[global_key].add(speaker_key)
                    committee = str(row.get("committee", "") or "").strip()
                    if committee:
                        global_neutral_committees[global_key].add(committee)
                    period = meeting_quarter(row.get("meeting_date", ""))
                    if period:
                        global_neutral_periods[global_key].add(period)
                    speaker_bloc = all_speaker_blocs.get(
                        (str(row.get("assembly_daesu", "")), speaker_key), ""
                    )
                    if speaker_bloc:
                        global_neutral_blocs[global_key].add(speaker_bloc)
            if target_type not in target_maps:
                continue
            target = row.get("target_name", "")
            slot = target_maps[target_type].get((election_id, target))
            if not slot:
                continue
            key = (election_id, text_hash, target_type, target)
            if not text_hash or key in seen:
                continue
            seen.add(key)
            cell_key = (election_id, slot, target_type)
            cell = totals[cell_key]
            cell["attention_count"] += 1
            issue_sets[cell_key].add(issue_name)
            if label in {"", "neutral", "ambiguous"}:
                if issue_name:
                    neutral_issue_counts[(*cell_key, issue_name)] += 1
                continue
            confidence = float(
                row.get("rule_stance_confidence")
                or row.get("stance_confidence")
                or 0.0
            )
            if confidence < min_confidence:
                continue
            if issue_name:
                directional_issue_sets[cell_key].add(issue_name)
            polarity = float(
                row.get("rule_stance_polarity")
                or row.get("stance_polarity")
                or 0.0
            )
            issue_weight = float(row.get("issue_weight") or 0.0)
            context_multiplier = 1.0
            if speaker_blocs:
                speaker_key = normalize_speaker_name(row.get("speaker", ""))
                speaker_bloc = speaker_blocs.get((str(row.get("assembly_daesu", "")), speaker_key), "")
                target_bloc = candidate_blocs.get((election_id, slot), "")
                if speaker_bloc and target_bloc:
                    cell["speaker_context_mapped_count"] += 1
                    if speaker_bloc == target_bloc:
                        cell["same_bloc_count"] += 1
                        context_multiplier = 1.20 if polarity < 0 else 1.00
                    else:
                        cell["other_bloc_count"] += 1
                        context_multiplier = 0.75 if polarity < 0 else 1.20
                else:
                    context_multiplier = 0.70
            confidence_weight = confidence**confidence_power / (0.65 ** (confidence_power - 1.0))
            weight = confidence_weight * issue_weight * context_multiplier
            cell["signed"] += polarity * weight
            cell["absolute"] += abs(polarity) * weight
            cell["evidence_count"] += 1
            if issue_name:
                directional_issue_mass[(*cell_key, issue_name)] += abs(weight)
                directional_issue_signed[(*cell_key, issue_name)] += polarity * weight
                directional_issue_absolute[(*cell_key, issue_name)] += abs(polarity) * weight

    global_issue_context: dict[tuple[str, str], dict[str, float]] = {}
    for global_key, count in global_neutral_counts.items():
        speaker_count = len(global_neutral_speakers[global_key])
        committee_count = len(global_neutral_committees[global_key])
        period_count = len(global_neutral_periods[global_key])
        bloc_count = len(global_neutral_blocs[global_key])
        structure_strength = (
            0.15 * bounded_log_strength(count, 500.0)
            + 0.25 * bounded_log_strength(speaker_count, 100.0)
            + 0.20 * bounded_log_strength(committee_count, 12.0)
            + 0.20 * bounded_log_strength(period_count, 12.0)
            + 0.20 * min(bloc_count / 2.0, 1.0)
        )
        content_strength = (
            0.60 * bounded_log_strength(global_neutral_analysis_counts[global_key], 80.0)
            + 0.40 * bounded_log_strength(global_neutral_impact_counts[global_key], 80.0)
        )
        global_issue_context[global_key] = {
            "neutral_count": float(count),
            "speaker_count": float(speaker_count),
            "committee_count": float(committee_count),
            "period_count": float(period_count),
            "bloc_count": float(bloc_count),
            "analysis_count": float(global_neutral_analysis_counts[global_key]),
            "impact_count": float(global_neutral_impact_counts[global_key]),
            "structure_strength": float(structure_strength),
            "content_strength": float(content_strength),
            "strength": float(0.75 * structure_strength + 0.25 * content_strength),
        }

    context_keys_by_election: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for global_key in global_issue_context:
        context_keys_by_election[global_key[0]].append(global_key)
    for context_keys in context_keys_by_election.values():
        strengths = pd.Series(
            [global_issue_context[key]["strength"] for key in context_keys],
            dtype=float,
        )
        if len(strengths) == 1:
            percentiles = np.ones(1, dtype=float)
        else:
            percentiles = ((strengths.rank(method="average") - 1.0) / (len(strengths) - 1.0)).to_numpy(float)
        for global_key, percentile in zip(context_keys, percentiles, strict=True):
            global_issue_context[global_key]["relative_strength"] = float(percentile**1.5)

    neutral_global_issue_gain = float(config.get("neutral_global_issue_gain", 0.0))
    neutral_global_issue_metric = str(config.get("neutral_global_issue_metric", "strength"))
    if neutral_global_issue_metric not in {"strength", "relative_strength"}:
        raise ValueError(f"unsupported neutral global issue metric: {neutral_global_issue_metric}")
    if neutral_global_issue_gain > 0.0:
        for issue_key, signed_value in directional_issue_signed.items():
            cell_key = issue_key[:3]
            issue_context = global_issue_context.get((issue_key[0], issue_key[3]))
            if not issue_context:
                continue
            extra = neutral_global_issue_gain * issue_context[neutral_global_issue_metric]
            totals[cell_key]["signed"] += signed_value * extra
            totals[cell_key]["absolute"] += directional_issue_absolute[issue_key] * extra

    def global_context_for_cell(cell_key: tuple[str, str, str]) -> dict[str, float]:
        issue_names = directional_issue_sets[cell_key]
        overlapping = [
            issue_name
            for issue_name in issue_names
            if (cell_key[0], issue_name) in global_issue_context
        ]
        result = {
            "neutral_count": 0.0,
            "speaker_count": 0.0,
            "committee_count": 0.0,
            "period_count": 0.0,
            "bloc_count": 0.0,
            "analysis_count": 0.0,
            "impact_count": 0.0,
            "issue_overlap_count": float(len(overlapping)),
            "structure_strength": 0.0,
            "content_strength": 0.0,
            "strength": 0.0,
            "relative_strength": 0.0,
        }
        if not overlapping:
            return result
        masses = np.asarray(
            [directional_issue_mass[(*cell_key, issue_name)] for issue_name in overlapping],
            dtype=float,
        )
        if masses.sum() <= 0.0:
            masses = np.ones(len(overlapping), dtype=float)
        for metric in (
            "neutral_count", "speaker_count", "committee_count", "period_count",
            "bloc_count", "analysis_count", "impact_count",
        ):
            result[metric] = float(
                sum(global_issue_context[(cell_key[0], issue_name)][metric] for issue_name in overlapping)
            )
        for metric in ("structure_strength", "content_strength", "strength", "relative_strength"):
            values = np.asarray(
                [global_issue_context[(cell_key[0], issue_name)][metric] for issue_name in overlapping],
                dtype=float,
            )
            result[metric] = float(np.average(values, weights=masses))
        return result

    out = candidates.copy()
    for target_type in ("person", "party"):
        keys = [(str(row.election_id), str(row.slot), target_type) for row in out.itertuples(index=False)]
        for column in (
            "signed", "absolute", "evidence_count", "attention_count",
            "speaker_context_mapped_count", "same_bloc_count", "other_bloc_count",
        ):
            out[f"{target_type}_{column}"] = [totals[key][column] for key in keys]
        out[f"{target_type}_issue_count"] = [len(issue_sets[key] - {""}) for key in keys]
        out[f"{target_type}_context_neutral_count"] = [
            sum(neutral_issue_counts[(*key, issue_name)] for issue_name in directional_issue_sets[key])
            for key in keys
        ]
        out[f"{target_type}_context_issue_overlap_count"] = [
            sum(neutral_issue_counts[(*key, issue_name)] > 0 for issue_name in directional_issue_sets[key])
            for key in keys
        ]
        global_context_rows = [global_context_for_cell(key) for key in keys]
        for metric in (
            "neutral_count", "speaker_count", "committee_count", "period_count",
            "bloc_count", "analysis_count", "impact_count", "issue_overlap_count",
            "structure_strength", "content_strength", "strength",
            "relative_strength",
        ):
            out[f"{target_type}_global_context_{metric}"] = [row[metric] for row in global_context_rows]
        net = out[f"{target_type}_signed"] / (out[f"{target_type}_absolute"] + 4.0)
        coverage = np.minimum(np.sqrt(out[f"{target_type}_evidence_count"]) / 8.0, 1.0)
        mention_strength = np.minimum(
            np.log1p(out[f"{target_type}_attention_count"]) / np.log1p(80.0),
            1.0,
        )
        issue_breadth = np.minimum(out[f"{target_type}_issue_count"] / 8.0, 1.0)
        attention_strength = 0.7 * mention_strength + 0.3 * issue_breadth
        out[f"{target_type}_signal"] = net * coverage * (1.0 + attention_gain * attention_strength)
    out["evidence_count"] = out["person_evidence_count"] + out["party_evidence_count"]
    out["attention_count"] = out["person_attention_count"] + out["party_attention_count"]
    out["context_neutral_count"] = (
        out["person_context_neutral_count"] + out["party_context_neutral_count"]
    )
    out["context_issue_overlap_count"] = (
        out["person_context_issue_overlap_count"] + out["party_context_issue_overlap_count"]
    )
    for metric in (
        "neutral_count", "speaker_count", "committee_count", "period_count",
        "bloc_count", "analysis_count", "impact_count", "issue_overlap_count",
    ):
        out[f"global_context_{metric}"] = (
            out[f"person_global_context_{metric}"] + out[f"party_global_context_{metric}"]
        )
    global_person_mass = out["person_absolute"].to_numpy(float)
    global_party_mass = party_weight * out["party_absolute"].to_numpy(float)
    global_total_mass = global_person_mass + global_party_mass
    for metric in ("structure_strength", "content_strength", "strength", "relative_strength"):
        weighted = (
            global_person_mass * out[f"person_global_context_{metric}"].to_numpy(float)
            + global_party_mass * out[f"party_global_context_{metric}"].to_numpy(float)
        )
        out[f"global_context_{metric}"] = np.divide(
            weighted,
            global_total_mass,
            out=np.zeros(len(out), dtype=float),
            where=global_total_mass > 0.0,
        )
    out["speaker_context_mapped_count"] = (
        out["person_speaker_context_mapped_count"] + out["party_speaker_context_mapped_count"]
    )
    out["same_bloc_count"] = out["person_same_bloc_count"] + out["party_same_bloc_count"]
    out["other_bloc_count"] = out["person_other_bloc_count"] + out["party_other_bloc_count"]
    out["covered_slot_count"] = out["evidence_count"].gt(0).groupby(out["election_id"]).transform("sum")
    out["coverage_gate_passed"] = out["covered_slot_count"].ge(min_covered_slots).astype(int)
    out["stance_net"] = out["person_signal"] + party_weight * out["party_signal"]
    if min_covered_slots > 0:
        out["stance_net"] *= out["coverage_gate_passed"]
    out["stance_shadow_signal"] = out["stance_net"]
    out["stance_shadow_signal"] -= out.groupby("election_id")["stance_shadow_signal"].transform("mean")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, float | str]] = []
    coverage_rows: list[dict[str, float | str]] = []
    prepared = shadow.prepare_2022_ab()

    for config in CONFIGS:
        features = build_features(config)
        for row in features.loc[features["election_id"].eq("pres_2022")].itertuples(index=False):
            coverage_rows.append({
                "config": str(config["name"]),
                "slot": str(row.slot),
                "candidate_name": str(row.candidate_name),
                "evidence_count": int(row.evidence_count),
                "attention_count": int(row.attention_count),
                "person_evidence_count": int(row.person_evidence_count),
                "party_evidence_count": int(row.party_evidence_count),
                "person_attention_count": int(row.person_attention_count),
                "party_attention_count": int(row.party_attention_count),
                "covered_slot_count": int(row.covered_slot_count),
                "coverage_gate_passed": int(row.coverage_gate_passed),
                "person_signal": float(row.person_signal),
                "party_signal": float(row.party_signal),
                "stance_net": float(row.stance_net),
                "stance_shadow_signal": float(row.stance_shadow_signal),
            })
        for scale in SCALES:
            metrics = shadow.run_2022_ab(features, scale=scale, prepared=prepared)
            result_rows.append({
                "config": str(config["name"]),
                "scale": scale,
                **metrics,
                "row_mae_reduction_pp": metrics["row_mae_baseline_pp"] - metrics["row_mae_shadow_pp"],
                "national_mae_reduction_pp": metrics["national_mae_baseline_pp"] - metrics["national_mae_shadow_pp"],
            })

    results = pd.DataFrame(result_rows)
    coverage = pd.DataFrame(coverage_rows)
    results.to_csv(OUTPUT_DIR / "sensitivity_results.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUTPUT_DIR / "feature_coverage.csv", index=False, encoding="utf-8-sig")

    best = results.sort_values(["row_mae_shadow_pp", "scale"]).iloc[0]
    best_by_config = results.loc[results.groupby("config")["row_mae_shadow_pp"].idxmin()].sort_values("row_mae_shadow_pp")
    best_by_config.to_csv(OUTPUT_DIR / "best_by_config.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 2022 3,000-Sentence Stance Sensitivity",
        "",
        "This is a stratified, rule-labelled, non-PIT diagnostic. It is not an active-model metric.",
        "",
        f"- baseline row MAE: `{best['row_mae_baseline_pp']:.4f}%p`",
        f"- lowest diagnostic row MAE: `{best['row_mae_shadow_pp']:.4f}%p`",
        f"- corresponding config: `{best['config']}`",
        f"- corresponding scale: `{best['scale']:.3f}`",
        f"- mean absolute prediction shift: `{best['mean_abs_shadow_shift_pp']:.4f}%p`",
        f"- search-boundary warning: `{best['scale'] == max(SCALES)}`",
        "- interpretation: the minimum occurs at the largest tested scale and is not a calibrated optimum",
        "- next-validation range: `0.060-0.120`; larger values are stress tests only",
        "",
        "## Results",
        "",
        results.to_csv(index=False),
        "",
        "## Best By Configuration",
        "",
        best_by_config.to_csv(index=False),
        "",
        "## Feature Coverage",
        "",
        coverage.to_csv(index=False),
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "rows": len(results),
        "baseline_row_mae_pp": float(best["row_mae_baseline_pp"]),
        "lowest_diagnostic_row_mae_pp": float(best["row_mae_shadow_pp"]),
        "config": str(best["config"]),
        "scale": float(best["scale"]),
        "mean_abs_shift_pp": float(best["mean_abs_shadow_shift_pp"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
