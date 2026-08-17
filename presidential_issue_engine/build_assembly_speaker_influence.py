"""Build assembly speaker influence and issue-conversion seed tables.

The builder uses already-extracted 15th Assembly issue phrase matches. It does
not reprocess long assembly transcript archives. The output is intentionally
conservative: it creates auditable inputs that can later be extended to 16th+
speaker-level reprocessing when those raw speaker rows are available.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.point_in_time import (  # noqa: E402
    cutoff_dates_as_strings,
    filter_observed_by_election,
)
from presidential_issue_engine.election_scope import ELECTION_DATES  # noqa: E402

DEFAULT_MATCHES = ROOT / "outputs/assembly_speaker_issue_matches_15_22.csv"
FALLBACK_15TH_MATCHES = ROOT / "outputs/15th_assembly_conversion/issue_phrase_extraction/15th_assembly_issue_phrase_matches.csv"
DEFAULT_ROSTER15 = ROOT / "data/raw/assembly15_member_roster.csv"
DEFAULT_ROSTER = ROOT / "data/assembly_roster.csv"
DEFAULT_MEMBER_HISTORY = ROOT / "data/raw/assembly_member_history.csv"
DEFAULT_PROFILE_OUT = ROOT / "data/raw/assembly_speaker_influence.csv"
DEFAULT_ISSUE_OUT = ROOT / "data/raw/assembly_issue_speaker_weighted.csv"
DEFAULT_SCOPE_OUT = ROOT / "data/raw/issue_scope_weights_assembly_derived.csv"
DEFAULT_CONVERSION_OUT = ROOT / "data/raw/issue_temporal_conversion.csv"
DEFAULT_DIAGNOSTICS_OUT = ROOT / "presidential_issue_engine/report/tables/assembly_speaker_influence_diagnostics.csv"

SIDO_ALIASES = {
    "서울": "sido_11",
    "부산": "sido_26",
    "대구": "sido_27",
    "인천": "sido_28",
    "광주": "sido_29",
    "대전": "sido_30",
    "울산": "sido_31",
    "경기": "sido_41",
    "강원": "sido_42",
    "충북": "sido_43",
    "충남": "sido_44",
    "전북": "sido_45",
    "전남": "sido_46",
    "경북": "sido_47",
    "경남": "sido_48",
    "제주": "sido_49",
    "세종": "sido_36",
}

CONSERVATIVE_HOME = {"sido_26", "sido_27", "sido_31", "sido_47", "sido_48"}
LIBERAL_HOME = {"sido_29", "sido_45", "sido_46"}
SWING_REGIONS = {"sido_11", "sido_28", "sido_30", "sido_36", "sido_41", "sido_42", "sido_43", "sido_44", "sido_49"}

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a UTF-8/CP949 CSV with string-preserving defaults."""

    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeError:
            continue
    return pd.read_csv(path, dtype=str)


def clean_speaker_name(value: object) -> str:
    """Normalize a transcript speaker label to a probable person/role token."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\([^)]*\)", "", text).strip()
    for token in ["의원", "위원장", "부위원장", "장관", "국무총리", "총리", "의장", "부의장", "대표"]:
        text = text.replace(token, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_role(speaker: object, committee: object = "", agenda: object = "") -> str:
    """Return a coarse institutional role from transcript labels."""

    text = " ".join([str(speaker or ""), str(committee or ""), str(agenda or "")])
    lower = text.lower()
    if "president" in lower:
        return "president"
    if "prime minister" in lower:
        return "prime_minister"
    if "minister" in lower or "cabinet" in lower:
        return "minister"
    if "assembly chair" in lower or "speaker of the assembly" in lower:
        return "assembly_chair"
    if "party leadership" in lower or "floor leader" in lower:
        return "party_leadership"
    if "committee chair" in lower:
        return "committee_chair"
    if "대통령" in text:
        return "president"
    if "국무총리" in text or re.search(r"(?<!부)총리", text):
        return "prime_minister"
    if "장관" in text or "차관" in text:
        return "minister"
    if "국회의장" in text or "의장" in text or "부의장" in text:
        return "assembly_chair"
    if "원내대표" in text or re.search(r"(?<!질문 )대표", text):
        return "party_leadership"
    if "위원장" in text:
        return "committee_chair"
    if "의원" in text:
        return "member"
    return "unknown"


def role_weight(role: str) -> float:
    """Weight formal political/institutional roles without making them dominant."""

    return {
        "president": 1.30,
        "prime_minister": 1.25,
        "minister": 1.15,
        "assembly_chair": 1.12,
        "party_leadership": 1.12,
        "committee_chair": 1.08,
        "member": 1.00,
        "unknown": 0.80,
    }.get(role, 0.85)


def meeting_weight(committee: object, agenda: object, source_sheet: object) -> float:
    """Weight meeting contexts by likely agenda visibility."""

    text = " ".join([str(committee or ""), str(agenda or ""), str(source_sheet or "")])
    lower = text.lower()
    weight = 1.0
    if "plenary" in lower or "본회의" in text:
        weight += 0.15
    if "question" in lower or "대정부" in text or "질문" in text:
        weight += 0.10
    if "audit" in lower or "국정감사" in text:
        weight += 0.05
    return min(weight, 1.35)


def normalize_member_id(value: object) -> str:
    """Normalize member IDs that may be serialized as floats."""

    text = str(value or "").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return re.sub(r"\.0$", "", text)


def normalize_daesu(value: object) -> str:
    """Normalize Assembly term labels such as '제21대' or '21.0' to digits."""

    text = str(value or "").strip()
    match = re.search(r"\d+", text)
    if not match:
        return ""
    return str(int(match.group(0)))


def district_sido(district: object) -> str:
    """Map a district string to a broad province code when possible."""

    text = str(district or "").strip()
    if not text:
        return ""
    for label, code in SIDO_ALIASES.items():
        if text.startswith(label) or label in text.split():
            return code
    # Common city-only 15th Assembly labels.
    if text.startswith(("마산", "창원", "진주", "김해", "통영")):
        return "sido_48"
    if text.startswith(("포항", "경주", "구미", "안동")):
        return "sido_47"
    if text.startswith(("전주", "군산", "익산")):
        return "sido_45"
    if text.startswith(("목포", "여수", "순천")):
        return "sido_46"
    if text.startswith(("청주", "충주")):
        return "sido_43"
    if text.startswith(("천안", "공주", "아산")):
        return "sido_44"
    return ""


def mandate_type(district: object, role: str) -> str:
    """Classify speaker mandate type."""

    text = str(district or "")
    lower = text.lower()
    if role in {"president", "prime_minister", "minister"}:
        return "government"
    if "proportional" in lower or "national list" in lower or "전국구" in text or "비례" in text:
        return "proportional"
    if text.strip():
        return "district"
    return "unknown"


def district_bloc_type(sido: str, bloc: object, mandate: str) -> str:
    """Classify district as home turf, hostile, swing, or national."""

    if mandate == "proportional":
        return "national_list"
    if mandate == "government":
        return "government"
    if not sido:
        return "unknown"
    bloc_text = str(bloc or "")
    if sido in SWING_REGIONS:
        return "swing"
    if sido in CONSERVATIVE_HOME:
        return "home_turf" if "국민의힘" in bloc_text else "hostile"
    if sido in LIBERAL_HOME:
        return "home_turf" if "더불어민주당" in bloc_text else "hostile"
    return "mixed"


def signal_weights(mandate: str, terrain: str) -> tuple[float, float, float, float]:
    """Return national/local/cross-region/base-mobilization speaker signal weights."""

    if mandate == "proportional":
        national, local = 0.85, 0.15
    elif mandate == "government":
        national, local = 0.90, 0.10
    elif mandate == "district":
        national, local = 0.40, 0.60
    else:
        national, local = 0.55, 0.45

    cross = 0.0
    base = 0.0
    if terrain == "hostile":
        cross = 0.35
        national += 0.05
    elif terrain == "swing":
        cross = 0.20
    elif terrain == "home_turf":
        base = 0.30
        local += 0.05

    total = national + local
    if total > 0:
        national, local = national / total, local / total
    return national, local, cross, base


def build_speaker_influence(
    matches: pd.DataFrame,
    roster15: pd.DataFrame,
    roster_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build speaker-level and issue-level influence tables."""

    frame = matches.copy()
    if "assembly_daesu" not in frame.columns:
        frame["assembly_daesu"] = ""
    if "election_id" not in frame.columns:
        frame["election_id"] = ""
    frame["election_id"] = frame["election_id"].replace("", "pres_2002").fillna("pres_2002")
    frame = filter_observed_by_election(
        frame,
        ELECTION_DATES,
        source_name="assembly_speaker_issue_matches",
        date_column="meeting_date",
    )
    frame["assembly_daesu"] = frame["assembly_daesu"].map(normalize_daesu)
    frame["member_id"] = frame["member_id"].map(normalize_member_id)
    frame["issue_weight"] = pd.to_numeric(frame["issue_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame["matched_term_count"] = pd.to_numeric(frame.get("matched_term_count", 1), errors="coerce").fillna(1.0)
    frame = frame.loc[frame["issue_weight"] > 0].copy()

    roster = roster15.copy()
    roster["member_id"] = roster["member_id"].map(normalize_member_id)
    roster["term_member_id"] = roster["term_member_id"].map(normalize_member_id)
    roster = roster.rename(
        columns={
            "name": "roster_name",
            "party": "speaker_party",
            "bloc": "speaker_bloc",
            "district": "district",
        }
    )
    by_member = roster[["member_id", "roster_name", "speaker_party", "speaker_bloc", "district"]].drop_duplicates(
        "member_id"
    )
    by_term_member = roster[
        ["term_member_id", "roster_name", "speaker_party", "speaker_bloc", "district"]
    ].drop_duplicates("term_member_id")
    frame = frame.merge(by_member, on="member_id", how="left")
    needs_term_lookup = frame["roster_name"].isna() & frame["member_id"].ne("")
    if needs_term_lookup.any():
        term_join = frame.loc[needs_term_lookup, ["member_id"]].rename(columns={"member_id": "term_member_id"})
        term_join = term_join.merge(by_term_member, on="term_member_id", how="left")
        for column in ["roster_name", "speaker_party", "speaker_bloc", "district"]:
            frame.loc[needs_term_lookup, column] = term_join[column].to_numpy()

    frame["speaker_clean"] = frame["speaker"].map(clean_speaker_name)
    roster_name_frames = [
        roster[["roster_name", "speaker_party", "speaker_bloc", "district"]].assign(assembly_daesu="15"),
    ]
    if not roster_all.empty and {"daesu", "name", "party", "bloc"}.issubset(roster_all.columns):
        all_roster = roster_all.copy()
        all_roster["assembly_daesu"] = all_roster["daesu"].map(normalize_daesu)
        if "district" not in all_roster.columns:
            all_roster["district"] = ""
        all_roster = all_roster.rename(
            columns={
                "name": "roster_name",
                "party": "speaker_party",
                "bloc": "speaker_bloc",
            }
        )
        roster_name_frames.append(
            all_roster[["assembly_daesu", "roster_name", "speaker_party", "speaker_bloc", "district"]]
        )
    by_name = pd.concat(roster_name_frames, ignore_index=True)
    by_name["speaker_clean"] = by_name["roster_name"].map(clean_speaker_name)
    by_name = by_name.dropna(subset=["speaker_clean"])
    by_name = by_name.loc[by_name["speaker_clean"].astype(str).str.len() > 0].copy()
    by_name = by_name.drop_duplicates(["assembly_daesu", "speaker_clean"])
    needs_name_lookup = frame["roster_name"].isna() & frame["speaker_clean"].astype(str).ne("")
    if needs_name_lookup.any():
        name_join = frame.loc[needs_name_lookup, ["assembly_daesu", "speaker_clean"]].merge(
            by_name,
            on=["assembly_daesu", "speaker_clean"],
            how="left",
        )
        for column in ["roster_name", "speaker_party", "speaker_bloc", "district"]:
            frame.loc[needs_name_lookup, column] = name_join[column].to_numpy()

    if not roster_all.empty:
        term_history = pd.concat(
            [
                roster15[["daesu", "name"]],
                roster_all[["daesu", "name"]],
            ],
            ignore_index=True,
        ).dropna()
        term_history["speaker_clean"] = term_history["name"].map(clean_speaker_name)
        term_history["term_daesu"] = pd.to_numeric(term_history["daesu"], errors="coerce")
        term_history = term_history.dropna(subset=["speaker_clean", "term_daesu"])
        term_history = term_history.drop_duplicates(["speaker_clean", "term_daesu"])
        terms_by_speaker = (
            term_history.groupby("speaker_clean")["term_daesu"]
            .apply(lambda values: tuple(sorted(set(values.astype(int)))))
            .to_dict()
        )
        speech_daesu = pd.to_numeric(frame["assembly_daesu"], errors="coerce")
        frame["term_count"] = [
            max(
                sum(term <= int(current_daesu) for term in terms_by_speaker.get(speaker, ())),
                1,
            )
            if pd.notna(current_daesu)
            else 1
            for speaker, current_daesu in zip(frame["speaker_clean"], speech_daesu)
        ]
    else:
        frame["term_count"] = 1
    frame["term_count"] = pd.to_numeric(frame["term_count"], errors="coerce").fillna(1).clip(lower=1)

    frame["institutional_role"] = [
        classify_role(speaker, committee, agenda)
        for speaker, committee, agenda in zip(frame["speaker"], frame.get("committee", ""), frame.get("agenda", ""))
    ]
    frame["mandate_type"] = [mandate_type(district, role) for district, role in zip(frame["district"], frame["institutional_role"])]
    frame["district_sido"] = frame["district"].map(district_sido)
    frame["district_bloc_type"] = [
        district_bloc_type(sido, bloc, mandate)
        for sido, bloc, mandate in zip(frame["district_sido"], frame["speaker_bloc"], frame["mandate_type"])
    ]

    frame["seniority_weight"] = (1.0 + (frame["term_count"] - 1.0).clip(lower=0.0, upper=4.0) * 0.05).clip(1.0, 1.20)
    frame["role_weight"] = frame["institutional_role"].map(role_weight).fillna(0.85)
    frame["meeting_weight"] = [
        meeting_weight(committee, agenda, source)
        for committee, agenda, source in zip(frame.get("committee", ""), frame.get("agenda", ""), frame.get("source_sheet", ""))
    ]
    signals = [signal_weights(mandate, terrain) for mandate, terrain in zip(frame["mandate_type"], frame["district_bloc_type"])]
    frame["national_signal_weight"] = [item[0] for item in signals]
    frame["local_signal_weight"] = [item[1] for item in signals]
    frame["cross_region_signal_weight"] = [item[2] for item in signals]
    frame["base_mobilization_weight"] = [item[3] for item in signals]
    frame["total_speech_weight"] = (
        frame["issue_weight"]
        * frame["matched_term_count"].clip(lower=1.0).pow(0.25)
        * frame["seniority_weight"]
        * frame["role_weight"]
        * frame["meeting_weight"]
    )
    speaker_issue_totals = (
        frame.groupby(["election_id", "issue_name", "speaker_clean"], as_index=False)["total_speech_weight"]
        .sum()
        .rename(columns={"total_speech_weight": "speaker_issue_total"})
    )
    speaker_issue_totals["speaker_issue_cap"] = speaker_issue_totals.groupby(
        ["election_id", "issue_name"]
    )["speaker_issue_total"].transform(
        lambda values: np.inf
        if len(values) < 8
        else max(float(values.quantile(0.90)), float(values.median()) * 2.5, 1e-9)
    )
    speaker_issue_totals["speaker_contribution_dampen"] = (
        speaker_issue_totals["speaker_issue_cap"] / speaker_issue_totals["speaker_issue_total"].replace(0.0, np.nan)
    ).clip(upper=1.0).fillna(1.0)
    frame = frame.merge(
        speaker_issue_totals[
            ["election_id", "issue_name", "speaker_clean", "speaker_contribution_dampen"]
        ],
        on=["election_id", "issue_name", "speaker_clean"],
        how="left",
    )
    frame["speaker_contribution_dampen"] = frame["speaker_contribution_dampen"].fillna(1.0)
    frame["total_speech_weight"] = frame["total_speech_weight"] * frame["speaker_contribution_dampen"]
    frame["mapping_confidence"] = np.select(
        [
            frame["roster_name"].notna(),
            frame["institutional_role"].isin(["president", "prime_minister", "minister", "assembly_chair"]),
        ],
        [0.90, 0.70],
        default=0.45,
    )
    frame["notes"] = np.where(
        frame["roster_name"].notna(),
        "speaker matched to Assembly roster/member history",
        "role inferred from transcript speaker label",
    )

    profile_columns = [
        "assembly_daesu",
        "election_id",
        "speaker_clean",
        "roster_name",
        "member_id",
        "speaker_party",
        "speaker_bloc",
        "mandate_type",
        "district",
        "district_sido",
        "district_bloc_type",
        "term_count",
        "seniority_weight",
        "institutional_role",
        "role_weight",
        "meeting_weight",
        "national_signal_weight",
        "local_signal_weight",
        "cross_region_signal_weight",
        "base_mobilization_weight",
        "mapping_confidence",
        "notes",
    ]
    speaker_profile = (
        frame.sort_values(["mapping_confidence", "total_speech_weight"], ascending=[False, False])
        .drop_duplicates(["assembly_daesu", "speaker_clean", "member_id", "institutional_role"])
        [profile_columns]
        .sort_values(["assembly_daesu", "speaker_clean", "member_id"])
        .reset_index(drop=True)
    )

    issue = frame.copy()
    issue["weighted_total"] = issue["total_speech_weight"]
    issue["national_signal"] = issue["weighted_total"] * issue["national_signal_weight"]
    issue["local_signal"] = issue["weighted_total"] * issue["local_signal_weight"]
    issue["cross_region_signal"] = issue["weighted_total"] * issue["cross_region_signal_weight"]
    issue["base_mobilization_signal"] = issue["weighted_total"] * issue["base_mobilization_weight"]
    def summarize(group_columns: list[str]) -> pd.DataFrame:
        summary = (
            issue.groupby(group_columns, as_index=False)
            .agg(
                weighted_total=("weighted_total", "sum"),
                national_signal=("national_signal", "sum"),
                local_signal=("local_signal", "sum"),
                cross_region_signal=("cross_region_signal", "sum"),
                base_mobilization_signal=("base_mobilization_signal", "sum"),
                matched_rows=("issue_name", "size"),
                unique_speakers=("speaker_clean", "nunique"),
                avg_seniority_weight=("seniority_weight", "mean"),
                avg_mapping_confidence=("mapping_confidence", "mean"),
                district_rows=("mandate_type", lambda values: int((values == "district").sum())),
                proportional_rows=("mandate_type", lambda values: int((values == "proportional").sum())),
                government_rows=("mandate_type", lambda values: int((values == "government").sum())),
            )
        )
        total = summary["weighted_total"].replace(0.0, np.nan)
        summary["district_speaker_share"] = summary["district_rows"] / summary["matched_rows"].clip(lower=1)
        summary["proportional_speaker_share"] = summary["proportional_rows"] / summary["matched_rows"].clip(lower=1)
        summary["government_speaker_share"] = summary["government_rows"] / summary["matched_rows"].clip(lower=1)
        summary["national_share"] = (summary["national_signal"] / total).fillna(0.5).clip(0.0, 1.0)
        summary["local_share"] = (summary["local_signal"] / total).fillna(0.5).clip(0.0, 1.0)
        summary["cross_region_share"] = (summary["cross_region_signal"] / total).fillna(0.0).clip(0.0, 1.0)
        summary["base_mobilization_share"] = (
            summary["base_mobilization_signal"] / total
        ).fillna(0.0).clip(0.0, 1.0)
        return summary

    issue_summary = summarize(["election_id", "issue_name"])
    scope_summary = summarize(["issue_name"])
    issue_summary["source"] = "assembly_15_22_speaker_weighting"
    issue_summary["notes"] = "Derived from 15th-22nd Assembly speaker-level issue phrase matches"

    scope = scope_summary[
        [
            "issue_name",
            "national_share",
            "local_share",
            "cross_region_share",
            "base_mobilization_share",
            "unique_speakers",
            "matched_rows",
            "avg_mapping_confidence",
        ]
    ].copy()
    scope["national_weight"] = scope["national_share"].clip(0.05, 0.95)
    scope["local_weight"] = scope["local_share"].clip(0.05, 0.95)
    norm = scope["national_weight"] + scope["local_weight"]
    scope["national_weight"] = scope["national_weight"] / norm
    scope["local_weight"] = scope["local_weight"] / norm
    scope["confidence"] = (
        0.30
        + 0.30 * (scope["unique_speakers"] / 25.0).clip(0.0, 1.0)
        + 0.25 * (scope["matched_rows"] / 100.0).clip(0.0, 1.0)
        + 0.15 * scope["avg_mapping_confidence"].clip(0.0, 1.0)
    ).clip(0.0, 0.85)
    scope["notes"] = "Speaker-role/district/proportional derived national/local split; diagnostic seed only"
    scope = scope[
        [
            "issue_name",
            "national_weight",
            "local_weight",
            "cross_region_share",
            "base_mobilization_share",
            "confidence",
            "notes",
        ]
    ]

    conversion = issue_summary[
        [
            "election_id",
            "issue_name",
            "national_share",
            "cross_region_share",
            "base_mobilization_share",
            "unique_speakers",
            "matched_rows",
            "avg_mapping_confidence",
        ]
    ].copy()
    conversion["temporal_sensitivity"] = (
        0.55
        + 0.25 * conversion["national_share"]
        + 0.20 * conversion["cross_region_share"]
    ).clip(0.35, 1.0)
    conversion["speaker_breadth"] = (
        0.55 * (conversion["unique_speakers"] / 50.0).clip(0.0, 1.0)
        + 0.45 * (conversion["matched_rows"] / 300.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    conversion["raw_conversion_multiplier"] = (
        1.0
        + 0.18 * (conversion["national_share"] - 0.50)
        + 0.55 * conversion["cross_region_share"]
        - 0.35 * conversion["base_mobilization_share"]
    )
    conversion["conversion_multiplier"] = (
        1.0
        + (conversion["raw_conversion_multiplier"] - 1.0)
        * (0.45 + 0.55 * conversion["speaker_breadth"])
    ).clip(0.85, 1.18)
    conversion["national_conversion_weight"] = conversion["national_share"].clip(0.0, 1.0)
    conversion["local_conversion_weight"] = (1.0 - conversion["national_conversion_weight"]).clip(0.0, 1.0)
    conversion["available_date"] = conversion["election_id"].map(cutoff_dates_as_strings(ELECTION_DATES)).fillna("")
    conversion["confidence"] = (
        0.25
        + 0.30 * (conversion["unique_speakers"] / 25.0).clip(0.0, 1.0)
        + 0.25 * (conversion["matched_rows"] / 100.0).clip(0.0, 1.0)
        + 0.10 * conversion["avg_mapping_confidence"].clip(0.0, 1.0)
    ).clip(0.0, 0.75)
    conversion["notes"] = (
        "15th-22nd Assembly speaker-role/district/proportional signal; low-strength issue conversion prior"
    )
    conversion = conversion[
        [
            "election_id",
            "issue_name",
            "temporal_sensitivity",
            "conversion_multiplier",
            "national_conversion_weight",
            "local_conversion_weight",
            "available_date",
            "confidence",
            "notes",
        ]
    ]

    diagnostics = pd.DataFrame(
        [
            {"metric": "match_rows", "value": len(frame)},
            {"metric": "unique_speakers", "value": frame["speaker_clean"].nunique()},
            {"metric": "roster_matched_rows", "value": int(frame["roster_name"].notna().sum())},
            {"metric": "district_rows", "value": int((frame["mandate_type"] == "district").sum())},
            {"metric": "proportional_rows", "value": int((frame["mandate_type"] == "proportional").sum())},
            {"metric": "government_rows", "value": int((frame["mandate_type"] == "government").sum())},
            {"metric": "issues", "value": issue_summary["issue_name"].nunique()},
        ]
    )

    return speaker_profile, issue_summary, scope, conversion, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--roster15", type=Path, default=DEFAULT_ROSTER15)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--member-history", type=Path, default=DEFAULT_MEMBER_HISTORY)
    parser.add_argument("--speaker-out", type=Path, default=DEFAULT_PROFILE_OUT)
    parser.add_argument("--issue-out", type=Path, default=DEFAULT_ISSUE_OUT)
    parser.add_argument("--scope-out", type=Path, default=DEFAULT_SCOPE_OUT)
    parser.add_argument("--conversion-out", type=Path, default=DEFAULT_CONVERSION_OUT)
    parser.add_argument("--diagnostics-out", type=Path, default=DEFAULT_DIAGNOSTICS_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches_path = args.matches if args.matches.exists() else FALLBACK_15TH_MATCHES
    matches = _read_csv(matches_path)
    roster15 = _read_csv(args.roster15)
    roster = _read_csv(args.roster) if args.roster.exists() else pd.DataFrame()
    if args.member_history.exists():
        member_history = _read_csv(args.member_history)
        roster = pd.concat([roster, member_history], ignore_index=True, sort=False)
    speaker_profile, issue_summary, scope, conversion, diagnostics = build_speaker_influence(
        matches,
        roster15,
        roster,
    )

    for path, frame in [
        (args.speaker_out, speaker_profile),
        (args.issue_out, issue_summary),
        (args.scope_out, scope),
        (args.conversion_out, conversion),
        (args.diagnostics_out, diagnostics),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        try:
            display_path = path.resolve().relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"saved {len(frame)} rows: {display_path}")


if __name__ == "__main__":
    main()
