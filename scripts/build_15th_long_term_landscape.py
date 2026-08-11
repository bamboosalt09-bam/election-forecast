"""Build long-term bloc landscape inputs from extracted 15th Assembly issue phrases."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from presidential_issue_engine.speech_landscape_builder import AXES, load_axis_map  # noqa: E402


MATCHES = ROOT / "outputs" / "15th_assembly_conversion" / "issue_phrase_extraction" / "15th_assembly_issue_phrase_matches.csv"
AXIS_MAP = ROOT / "data" / "raw" / "political_landscape_issue_axis.csv"
OUT_ROSTER = ROOT / "data" / "raw" / "assembly15_member_roster.csv"
OUT_ISSUE = ROOT / "data" / "raw" / "assembly15_bloc_issue_profile.csv"
OUT_LANDSCAPE = ROOT / "data" / "raw" / "assembly15_bloc_political_landscape.csv"


PARTY_TO_BLOC = {
    "신한국당": "국민의힘",
    "한나라당": "국민의힘",
    "민주자유당": "국민의힘",
    "자유민주연합": "제3지대",
    "자민련": "제3지대",
    "새정치국민회의": "더불어민주당",
    "새천년민주당": "더불어민주당",
    "통합민주당": "더불어민주당",
    "국민회의": "더불어민주당",
    "민주당": "더불어민주당",
    "무소속": "무소속",
}

ROLE_TOKENS = [
    "위원장",
    "부위원장",
    "간사",
    "의원",
    "의장",
    "부의장",
    "국무총리",
    "부총리겸재정경제원장관",
    "부총리겸통일원장관",
    "재정경제부장관",
    "법무부장관",
    "국방부장관",
    "통일부장관",
    "외교통상부장관",
    "노동부장관",
    "여성특별위원장",
]


def clean_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    for token in ROLE_TOKENS:
        text = text.replace(token, " ")
    text = " ".join(text.split())
    return text.split()[-1] if text.split() else text


def party_to_bloc(party: object) -> str:
    text = "" if pd.isna(party) else str(party).strip()
    compact = text.replace(" ", "")
    if compact in PARTY_TO_BLOC:
        return PARTY_TO_BLOC[compact]
    has_conservative = any(token in compact for token in ["신한국당", "한나라당", "민주자유당", "민자당"])
    has_liberal = any(token in compact for token in ["새정치국민회의", "새천년민주당", "국민회의", "민주당"])
    has_third = any(token in compact for token in ["자유민주연합", "자민련"])
    if compact == "무소속":
        return "무소속"
    active = sum(bool(value) for value in [has_conservative, has_liberal, has_third])
    if active > 1:
        return "mixed"
    if has_conservative:
        return "국민의힘"
    if has_liberal:
        return "더불어민주당"
    if has_third:
        return "제3지대"
    return "unmapped"


def load_15th_roster_from_zip() -> pd.DataFrame:
    zip_path = Path.home() / "Downloads" / "trash_dataset.zip"
    with zipfile.ZipFile(zip_path) as archive:
        xlsx_info = next(info for info in archive.infolist() if info.filename.lower().endswith(".xlsx"))
        data = archive.read(xlsx_info)
    roster = pd.read_excel(io.BytesIO(data), dtype=str)
    frame = roster.loc[roster["대수"].astype(str) == "15"].copy()
    frame = frame.rename(
        columns={
            "대수": "daesu",
            "이름": "name",
            "대수별 의원ID": "term_member_id",
            "정당": "party",
            "지역구": "district",
            "의원ID": "member_id",
        }
    )
    frame["bloc"] = frame["party"].map(party_to_bloc)
    return frame[["daesu", "name", "term_member_id", "party", "bloc", "district", "member_id"]]


def attach_roster(matches: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    out = matches.copy()
    out["speaker_clean"] = out["speaker"].map(clean_name)
    out["member_id"] = out["member_id"].astype(str).str.strip()
    roster_by_term_id = roster.drop_duplicates("term_member_id")[
        ["term_member_id", "name", "party", "bloc"]
    ].rename(columns={"term_member_id": "member_id", "name": "roster_name"})
    out = out.merge(roster_by_term_id, on="member_id", how="left")
    missing = out["bloc"].isna()
    roster_by_name = roster.drop_duplicates("name")[["name", "party", "bloc"]].rename(
        columns={"name": "speaker_clean", "party": "party_by_name", "bloc": "bloc_by_name"}
    )
    out = out.merge(roster_by_name, on="speaker_clean", how="left")
    out.loc[missing, "party"] = out.loc[missing, "party_by_name"]
    out.loc[missing, "bloc"] = out.loc[missing, "bloc_by_name"]
    out["bloc"] = out["bloc"].fillna("unmapped")
    out["party"] = out["party"].fillna("")
    return out


def build_issue_profile(joined: pd.DataFrame) -> pd.DataFrame:
    usable = joined.loc[~joined["bloc"].isin(["unmapped", "mixed"])].copy()
    usable["issue_weight"] = pd.to_numeric(usable["issue_weight"], errors="coerce").fillna(0.0)
    grouped = (
        usable.groupby(["bloc", "party", "issue_name"], as_index=False)
        .agg(
            matched_rows=("issue_name", "size"),
            issue_weight_sum=("issue_weight", "sum"),
            unique_speakers=("speaker_clean", "nunique"),
        )
        .sort_values(["bloc", "party", "issue_weight_sum"], ascending=[True, True, False])
    )
    total = grouped.groupby(["bloc", "party"])["issue_weight_sum"].transform("sum").replace(0.0, np.nan)
    grouped["emphasis_within"] = (grouped["issue_weight_sum"] / total).fillna(0.0)
    grouped["source"] = "15th_assembly_issue_phrase_extraction"
    return grouped


def build_bloc_landscape(issue_profile: pd.DataFrame, axis_map: pd.DataFrame) -> pd.DataFrame:
    profile = (
        issue_profile.groupby(["bloc", "issue_name"], as_index=False)
        .agg(issue_weight_sum=("issue_weight_sum", "sum"), matched_rows=("matched_rows", "sum"))
    )
    total = profile.groupby("bloc")["issue_weight_sum"].transform("sum").replace(0.0, np.nan)
    profile["emphasis_within"] = (profile["issue_weight_sum"] / total).fillna(0.0)
    joined = profile.merge(axis_map[["issue_name", *AXES]], on="issue_name", how="inner")
    for axis in AXES:
        joined[axis] = joined[axis] * joined["emphasis_within"]
    vectors = joined.groupby("bloc", as_index=False)[AXES].sum()
    evidence = profile.groupby("bloc", as_index=False).agg(
        matched_rows=("matched_rows", "sum"),
        issue_count=("issue_name", "nunique"),
    )
    vectors = vectors.merge(evidence, on="bloc", how="left")
    vectors["available_date"] = "2000-01-15"
    vectors["confidence"] = (
        np.sqrt(pd.to_numeric(vectors["matched_rows"], errors="coerce").fillna(0.0))
        / (
            np.sqrt(pd.to_numeric(vectors["matched_rows"], errors="coerce").fillna(0.0))
            + 25.0
        )
    ).clip(0.0, 0.75)
    vectors["source"] = "15th_assembly_issue_phrase_extraction"
    vectors["notes"] = "Derived from converted 15th National Assembly xlsx/PDF issue phrase matches; unmapped speakers excluded"
    return vectors[
        [
            "bloc",
            *AXES,
            "matched_rows",
            "issue_count",
            "available_date",
            "confidence",
            "source",
            "notes",
        ]
    ].sort_values("bloc", ignore_index=True)


def main() -> None:
    OUT_ROSTER.parent.mkdir(parents=True, exist_ok=True)
    roster = load_15th_roster_from_zip()
    matches = pd.read_csv(MATCHES)
    joined = attach_roster(matches, roster)
    issue_profile = build_issue_profile(joined)
    landscape = build_bloc_landscape(issue_profile, load_axis_map(AXIS_MAP))
    roster.to_csv(OUT_ROSTER, index=False, encoding="utf-8-sig")
    issue_profile.to_csv(OUT_ISSUE, index=False, encoding="utf-8-sig")
    landscape.to_csv(OUT_LANDSCAPE, index=False, encoding="utf-8-sig")
    print(f"saved {len(roster)} roster rows: {OUT_ROSTER}")
    print(f"saved {len(issue_profile)} issue rows: {OUT_ISSUE}")
    print(f"saved {len(landscape)} bloc landscape rows: {OUT_LANDSCAPE}")
    print(landscape[["bloc", "matched_rows", "issue_count", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
