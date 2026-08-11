"""Extract speaker-level issue matches for the 16th-22nd Assemblies.

This script streams the user-provided Assembly transcript workbooks and writes
matched issue rows that can be combined with the already-extracted 15th
Assembly rows. It deliberately preserves speaker/date/source metadata so the
speaker-influence builder can weight seniority, office, district/proportional
status, hostile districts, and home-turf signals.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.features.issue_matcher import match_issue_weights  # noqa: E402
from news_collector.sources.assembly_batch import ELECTION_WINDOWS, iter_rows_from_xlsx  # noqa: E402
from news_collector.sources.assembly_batch import _SPEECH_COLS  # noqa: E402
from news_collector.sources.assembly_records import _week, parse_meeting_date  # noqa: E402
from news_collector.sources.contextual_issue_weights import (  # noqa: E402
    housing_issue_boosts_from_index,
    load_issue_context_rules,
)
from news_collector.sources.datalab import load_issue_keywords  # noqa: E402
from news_collector.sources.issue_term_weights import load_campaign_issue_terms, merge_issue_terms  # noqa: E402
from news_collector.sources.member_party import party_bloc  # noqa: E402

DOWNLOADS = Path.home() / "Downloads"
DEFAULT_SOURCE_DIR = DOWNLOADS / "trash_dataset"
DEFAULT_SOURCE_ZIP = DOWNLOADS / "trash_dataset.zip"
DEFAULT_15TH_MATCHES = (
    ROOT
    / "outputs/15th_assembly_conversion/issue_phrase_extraction/15th_assembly_issue_phrase_matches.csv"
)
DEFAULT_OUT = ROOT / "outputs/assembly_speaker_issue_matches_16_22.csv"
DEFAULT_COMBINED_OUT = ROOT / "outputs/assembly_speaker_issue_matches_15_22.csv"
DEFAULT_HISTORY_SOURCE = DOWNLOADS / "\uc5ed\ub300\uad6d\ud68c\uc758\uc6d0\uc778\uc801\uc0ac\ud56d.csv"
DEFAULT_HISTORY_OUT = ROOT / "data/raw/assembly_member_history.csv"

KEYWORDS = ROOT / "presidential_issue_engine/fixed_dataset/issue_keywords.csv"
CAMPAIGN_TERMS = ROOT / "presidential_issue_engine/fixed_dataset/campaign_issue_terms.csv"
MEGA_TERMS = ROOT / "presidential_issue_engine/fixed_dataset/mega_issue_terms.csv"
CONTEXT_RULES = ROOT / "presidential_issue_engine/fixed_dataset/issue_context_rules.csv"
HOUSING_INDEX = ROOT / "presidential_issue_engine/fixed_dataset/housing_price_index_sido.csv"

DATE_COL = "\ud68c\uc758\uc77c\uc790"
COMMITTEE_COL = "\uc704\uc6d0\ud68c"
AGENDA_COL = "\uc548\uac74"
SPEAKER_COL = "\ubc1c\uc5b8\uc790"
MEMBER_ID_COL = "\uc758\uc6d0ID"

ELECTION_DATES = {
    "pres_2002": pd.Timestamp("2002-12-19"),
    "pres_2007": pd.Timestamp("2007-12-19"),
    "pres_2012": pd.Timestamp("2012-12-19"),
    "pres_2017": pd.Timestamp("2017-05-09"),
    "pres_2022": pd.Timestamp("2022-03-09"),
}
ELECTION_CUTOFFS = {
    "pres_2002": (pd.Timestamp("1996-01-01"), pd.Timestamp("2002-12-18")),
    "pres_2007": (pd.Timestamp("2002-12-20"), pd.Timestamp("2007-12-18")),
    "pres_2012": (pd.Timestamp("2007-12-20"), pd.Timestamp("2012-12-18")),
    "pres_2017": (pd.Timestamp("2012-12-20"), pd.Timestamp("2017-05-08")),
    "pres_2022": (pd.Timestamp("2017-05-10"), pd.Timestamp("2022-03-08")),
}
ELECTION_IDS = tuple(ELECTION_DATES)
ELECTION_TO_ASSEMBLY = {
    "pres_2002": "16",
    "pres_2007": "17",
    "pres_2012": "18",
    "pres_2017": "20",
    "pres_2022": "21",
}

OUTPUT_COLUMNS = [
    "election_id",
    "assembly_daesu",
    "source_sheet",
    "source_file",
    "source_row_id",
    "meeting_date",
    "period",
    "committee",
    "agenda",
    "speaker",
    "member_id",
    "issue_name",
    "issue_weight",
    "matched_term_count",
    "text_length",
]


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeError:
            continue
    return pd.read_csv(path, dtype=str)


def assembly_daesu_from_name(name: str) -> str:
    match = re.search(r"(\d{2})", name)
    return str(int(match.group(1))) if match else ""


def election_for_date(value: object, window_mode: str = "continuous") -> str:
    d = parse_meeting_date(value)
    if d is None:
        return ""
    ts = pd.Timestamp(d)
    if window_mode == "campaign":
        iso = d.isoformat()
        for election_id, (start, end) in ELECTION_WINDOWS.items():
            if start <= iso <= end:
                return election_id
        return ""
    for election_id, (start, end) in ELECTION_CUTOFFS.items():
        if start <= ts <= end:
            return election_id
    return ""


def iter_workbooks_from_source(source: Path) -> Iterable[tuple[str, bytes]]:
    """Yield workbook bytes from a directory, a zip, or a direct xlsx."""

    if source.is_dir():
        for path in sorted(source.iterdir(), key=lambda p: p.name):
            if path.is_dir():
                continue
            lower = path.name.lower()
            if lower.endswith(".xlsx"):
                yield path.name, path.read_bytes()
            elif lower.endswith(".zip"):
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if info.is_dir() or not info.filename.lower().endswith(".xlsx"):
                            continue
                        yield f"{path.name}::{info.filename}", archive.read(info)
        return

    if source.name.lower().endswith(".xlsx"):
        yield source.name, source.read_bytes()
        return

    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.filename.lower().endswith(".xlsx"):
                yield info.filename, archive.read(info)
            elif info.filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(archive.read(info))) as inner:
                    for nested in inner.infolist():
                        if nested.is_dir() or not nested.filename.lower().endswith(".xlsx"):
                            continue
                        yield f"{info.filename}::{nested.filename}", inner.read(nested)


def build_keyword_inputs() -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, dict[tuple[str, str], float]],
    dict[str, dict[str, float]],
    dict[str, list[object]],
]:
    base_keywords = load_issue_keywords(KEYWORDS)
    campaign_terms, campaign_weights = load_campaign_issue_terms(CAMPAIGN_TERMS, ELECTION_IDS)
    mega_terms, mega_weights = load_campaign_issue_terms(MEGA_TERMS, ELECTION_IDS)
    issue_boosts, _ = housing_issue_boosts_from_index(
        HOUSING_INDEX,
        {key: value.strftime("%Y-%m-%d") for key, value in ELECTION_DATES.items()},
        ELECTION_TO_ASSEMBLY,
    )
    context_rules = load_issue_context_rules(CONTEXT_RULES, ELECTION_IDS)

    keyword_maps: dict[str, dict[str, list[str]]] = {}
    term_weights: dict[str, dict[tuple[str, str], float]] = {}
    for election_id in ELECTION_IDS:
        keyword_maps[election_id] = merge_issue_terms(
            merge_issue_terms(base_keywords, campaign_terms.get(election_id, {})),
            mega_terms.get(election_id, {}),
        )
        term_weights[election_id] = {
            **campaign_weights.get(election_id, {}),
            **mega_weights.get(election_id, {}),
        }
    return keyword_maps, term_weights, issue_boosts, context_rules


def completed_source_files(out_path: Path) -> set[str]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(out_path, usecols=["source_file"], dtype=str)
    except Exception:  # noqa: BLE001 - corrupt/incomplete resume file should not hide work
        return set()
    return set(frame["source_file"].dropna().astype(str).unique())


def extract_matches(
    source: Path,
    out_path: Path,
    max_workbooks: int | None = None,
    resume: bool = True,
    window_mode: str = "continuous",
) -> None:
    keyword_maps, term_weights, issue_boosts, context_rules = build_keyword_inputs()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_sources = completed_source_files(out_path) if resume else set()
    write_header = not (resume and out_path.exists() and out_path.stat().st_size > 0)
    mode = "a" if resume and out_path.exists() and out_path.stat().st_size > 0 else "w"
    total_rows = 0
    total_matches = 0
    with out_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        for workbook_idx, (name, data) in enumerate(iter_workbooks_from_source(source), 1):
            if max_workbooks is not None and workbook_idx > max_workbooks:
                break
            if name in done_sources:
                print(f"[{workbook_idx}] skip-completed {name[:100]}", flush=True)
                continue
            assembly_daesu = assembly_daesu_from_name(name)
            workbook_rows = 0
            workbook_matches = 0
            try:
                for row_idx, row in enumerate(iter_rows_from_xlsx(data), 1):
                    workbook_rows += 1
                    election_id = election_for_date(row.get(DATE_COL), window_mode=window_mode)
                    if not election_id:
                        continue
                    text = " ".join(str(row.get(column) or "") for column in _SPEECH_COLS if row.get(column))
                    if not text.strip():
                        continue
                    issue_weights = match_issue_weights(
                        text,
                        keyword_maps[election_id],
                        term_weights=term_weights.get(election_id),
                        issue_boosts=issue_boosts.get(election_id),
                        context_rules=context_rules.get(election_id),
                    )
                    if not issue_weights:
                        continue
                    d = parse_meeting_date(row.get(DATE_COL))
                    if d is None:
                        continue
                    for issue_name, weight in issue_weights.items():
                        writer.writerow(
                            {
                                "election_id": election_id,
                                "assembly_daesu": assembly_daesu,
                                "source_sheet": "",
                                "source_file": name,
                                "source_row_id": row_idx,
                                "meeting_date": d.isoformat(),
                                "period": _week(d),
                                "committee": str(row.get(COMMITTEE_COL) or ""),
                                "agenda": str(row.get(AGENDA_COL) or ""),
                                "speaker": str(row.get(SPEAKER_COL) or ""),
                                "member_id": str(row.get(MEMBER_ID_COL) or ""),
                                "issue_name": issue_name,
                                "issue_weight": float(weight),
                                "matched_term_count": 1,
                                "text_length": len(text),
                            }
                        )
                        workbook_matches += 1
                total_rows += workbook_rows
                total_matches += workbook_matches
                print(
                    f"[{workbook_idx}] {name[:100]} rows={workbook_rows} "
                    f"matches={workbook_matches} total_matches={total_matches}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - keep long extraction moving
                print(f"[{workbook_idx}] skip {name[:100]} error={exc!r}", flush=True)
    action = "appended" if mode == "a" else "saved"
    print(f"{action} {total_matches} match rows from {total_rows} speech rows: {out_path}", flush=True)


def normalize_member_history(source: Path, out_path: Path) -> pd.DataFrame:
    if not source.exists():
        return pd.DataFrame()
    frame = _read_csv(source).fillna("")
    column_map = {
        "\ub300": "daesu_label",
        "\uc774\ub984": "name",
        "\uc815\ub2f9\uba85": "party",
        "\uc120\uac70\uad6c": "district",
        "\uc120\uac70\uad6c\uad6c\ubd84": "mandate_label",
    }
    missing = [column for column in column_map if column not in frame.columns]
    if missing:
        return pd.DataFrame()
    out = frame[list(column_map)].rename(columns=column_map).copy()
    out["daesu"] = out["daesu_label"].astype(str).str.extract(r"(\d+)")[0].fillna("")
    out["bloc"] = out["party"].map(party_bloc)
    out = out[["daesu", "name", "party", "bloc", "district", "mandate_label"]]
    out = out.drop_duplicates(["daesu", "name", "party", "district", "mandate_label"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} member-history rows: {out_path}", flush=True)
    return out


def combine_with_15th(matches_15: Path, matches_16_22: Path, out_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    if matches_15.exists():
        frame15 = _read_csv(matches_15)
        frame15["assembly_daesu"] = "15"
        if "election_id" not in frame15.columns:
            frame15["election_id"] = ""
        frame15["election_id"] = frame15.apply(
            lambda row: row["election_id"] or election_for_date(row.get("meeting_date")) or "pres_2002",
            axis=1,
        )
        for column in OUTPUT_COLUMNS:
            if column not in frame15.columns:
                frame15[column] = ""
        frames.append(frame15[OUTPUT_COLUMNS])
    if matches_16_22.exists():
        frame = _read_csv(matches_16_22)
        for column in OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        frames.append(frame[OUTPUT_COLUMNS])
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {len(combined)} combined match rows: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = DEFAULT_SOURCE_DIR if DEFAULT_SOURCE_DIR.exists() else DEFAULT_SOURCE_ZIP
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--matches-15", type=Path, default=DEFAULT_15TH_MATCHES)
    parser.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED_OUT)
    parser.add_argument("--member-history-source", type=Path, default=DEFAULT_HISTORY_SOURCE)
    parser.add_argument("--member-history-out", type=Path, default=DEFAULT_HISTORY_OUT)
    parser.add_argument("--max-workbooks", type=int, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--window-mode", choices=["continuous", "campaign"], default="continuous")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_member_history(args.member_history_source, args.member_history_out)
    if not args.skip_extract:
        extract_matches(
            args.source,
            args.out,
            args.max_workbooks,
            resume=not args.no_resume,
            window_mode=args.window_mode,
        )
    combine_with_15th(args.matches_15, args.out, args.combined_out)


if __name__ == "__main__":
    main()
