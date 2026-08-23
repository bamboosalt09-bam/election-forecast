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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
DEFAULT_2025_SUPPLEMENT = (
    ROOT
    / "data/raw/official_sources/assembly_pres_2025_minutes/assembly_stance_rows_2025_h1.csv"
)
#: The redistributable derivation of the file above, used when the collected one
#: is absent. It carries text_length instead of text_excerpt, which is the only
#: thing this module takes from the excerpts - see
#: scripts/build_redistributable_pres_2025_stance_rows.py. Without it the 2025
#: demonstration cannot be rebuilt from a clean checkout at all.
PUBLIC_2025_SUPPLEMENT = (
    ROOT
    / "data/raw/official_sources/assembly_pres_2025_minutes/assembly_stance_rows_2025_h1_public.csv.gz"
)


def resolve_2025_supplement(source: Path) -> Path:
    """Prefer the collected rows; fall back to their redistributable form."""

    if source.exists():
        return source
    if source == DEFAULT_2025_SUPPLEMENT and PUBLIC_2025_SUPPLEMENT.exists():
        return PUBLIC_2025_SUPPLEMENT
    return source
DEFAULT_2025_OUT = ROOT / "outputs/assembly_speaker_issue_matches_pres_2025.csv"

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
    "pres_2025": pd.Timestamp("2025-06-03"),
}
ELECTION_CUTOFFS = {
    "pres_2002": (pd.Timestamp("1996-01-01"), pd.Timestamp("2002-12-18")),
    "pres_2007": (pd.Timestamp("2002-12-20"), pd.Timestamp("2007-12-18")),
    "pres_2012": (pd.Timestamp("2007-12-20"), pd.Timestamp("2012-12-18")),
    "pres_2017": (pd.Timestamp("2012-12-20"), pd.Timestamp("2017-05-08")),
    "pres_2022": (pd.Timestamp("2017-05-10"), pd.Timestamp("2022-03-08")),
    "pres_2025": (pd.Timestamp("2022-03-10"), pd.Timestamp("2025-06-02")),
}
ELECTION_IDS = tuple(ELECTION_DATES)
ELECTION_TO_ASSEMBLY = {
    "pres_2002": "16",
    "pres_2007": "17",
    "pres_2012": "19",
    "pres_2017": "20",
    "pres_2022": "21",
    "pres_2025": "21|22",
}
MODEL_ASSEMBLIES = frozenset(
    assembly
    for value in ELECTION_TO_ASSEMBLY.values()
    for assembly in value.split("|")
)

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

FORBIDDEN_OUTCOME_COLUMN_TOKENS = (
    "actual",
    "vote_share",
    "votes",
    "rank",
    "winner",
    "elected",
    "result",
    "\ub4dd\ud45c",
    "\ub2f9\uc120",
    "\uc21c\uc704",
)


def _text_length(selected: pd.DataFrame) -> pd.Series:
    """Character count of each excerpt, however the source supplies it.

    The collected rows carry the excerpt; the redistributable derivation carries
    only its length. Nothing downstream reads the text itself.
    """

    if "text_excerpt" in selected.columns:
        return selected["text_excerpt"].astype(str).str.len()
    if "text_length" in selected.columns:
        return pd.to_numeric(selected["text_length"], errors="coerce").fillna(0).astype(int)
    raise ValueError(
        "the 2025 supplement carries neither text_excerpt nor text_length"
    )



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


def assembly_allowed_for_election(assembly_daesu: str, election_id: str) -> bool:
    allowed = set(ELECTION_TO_ASSEMBLY.get(election_id, "").split("|"))
    return bool(assembly_daesu and assembly_daesu in allowed)


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


def normalized_source_file(value: object) -> str:
    """Remove only the outer archive-directory prefix from a workbook key."""

    text = str(value or "").replace("\\", "/")
    if text.startswith("trash_dataset/"):
        return text[len("trash_dataset/") :]
    return text


def completed_source_files(out_path: Path) -> set[str]:
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(out_path, usecols=["source_file"], dtype=str)
    except Exception:  # noqa: BLE001 - corrupt/incomplete resume file should not hide work
        return set()
    return {
        normalized_source_file(value)
        for value in frame["source_file"].dropna().astype(str).unique()
    }


def extract_matches(
    source: Path,
    out_path: Path,
    max_workbooks: int | None = None,
    resume: bool = True,
    window_mode: str = "campaign",
    assemblies: set[str] | None = None,
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
            source_name = normalized_source_file(name)
            if max_workbooks is not None and workbook_idx > max_workbooks:
                break
            if source_name in done_sources:
                print(f"[{workbook_idx}] skip-completed {name[:100]}", flush=True)
                continue
            assembly_daesu = assembly_daesu_from_name(source_name)
            if assemblies and assembly_daesu not in assemblies:
                print(
                    f"[{workbook_idx}] skip-unrequested-assembly {name[:100]}",
                    flush=True,
                )
                continue
            if assembly_daesu and assembly_daesu not in MODEL_ASSEMBLIES:
                print(
                    f"[{workbook_idx}] skip-out-of-scope-assembly {name[:100]}",
                    flush=True,
                )
                continue
            workbook_rows = 0
            workbook_matches = 0
            workbook_output: list[dict[str, object]] = []
            try:
                for row_idx, row in enumerate(iter_rows_from_xlsx(data), 1):
                    workbook_rows += 1
                    election_id = election_for_date(row.get(DATE_COL), window_mode=window_mode)
                    if not election_id or not assembly_allowed_for_election(
                        assembly_daesu,
                        election_id,
                    ):
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
                        workbook_output.append(
                            {
                                "election_id": election_id,
                                "assembly_daesu": assembly_daesu,
                                "source_sheet": "",
                                "source_file": source_name,
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
                writer.writerows(workbook_output)
                handle.flush()
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


def convert_pres_2025_official_supplement(
    source: Path,
    out_path: Path,
    window_mode: str = "campaign",
) -> pd.DataFrame:
    """Convert the official 2025 minutes supplement to the historical match schema.

    The official collector already ran the same issue matcher at sentence level.
    This adapter only applies the central D-1 availability rule and reshapes its
    output; it does not infer or hand-enter any issue or candidate signal.
    """

    source = resolve_2025_supplement(source)
    if not source.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = _read_csv(source).fillna("")
    lowered = {str(column).strip().lower() for column in frame.columns}
    forbidden = sorted(
        column
        for column in lowered
        if any(token in column for token in FORBIDDEN_OUTCOME_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"2025 official supplement contains forbidden outcome columns: {forbidden}")

    required = {
        "election_id",
        "assembly_daesu",
        "source_id",
        "source_file",
        "source_row_id",
        "sentence_index",
        "meeting_date",
        "available_date",
        "period",
        "committee",
        "agenda",
        "speaker",
        "member_id",
        "issue_name",
        "issue_weight",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"2025 official supplement is missing columns: {missing}")

    meeting_date = pd.to_datetime(frame["meeting_date"], errors="coerce")
    available_date = pd.to_datetime(frame["available_date"], errors="coerce")
    cutoff = ELECTION_DATES["pres_2025"] - pd.Timedelta(days=1)
    eligible = (
        frame["election_id"].eq("pres_2025")
        & frame["assembly_daesu"].astype(str).eq("22")
        & meeting_date.notna()
        & available_date.notna()
        & available_date.le(cutoff)
    )
    if window_mode == "campaign":
        start, end = ELECTION_WINDOWS["pres_2025"]
        eligible &= meeting_date.between(pd.Timestamp(start), pd.Timestamp(end))
    else:
        start, end = ELECTION_CUTOFFS["pres_2025"]
        eligible &= meeting_date.between(start, end)

    selected = frame.loc[eligible].copy()
    if selected.empty:
        raise ValueError(
            "2025 official supplement produced no point-in-time eligible issue rows; "
            "check its meeting_date and available_date coverage"
        )
    selected["issue_weight"] = pd.to_numeric(selected["issue_weight"], errors="coerce")
    selected = selected.loc[selected["issue_weight"].notna() & selected["issue_weight"].gt(0)].copy()

    out = pd.DataFrame(
        {
            "election_id": "pres_2025",
            "assembly_daesu": "22",
            "source_sheet": "official_minutes",
            "source_file": selected["source_file"].astype(str),
            "source_row_id": (
                selected["source_id"].astype(str)
                + ":"
                + selected["source_row_id"].astype(str)
                + ":"
                + selected["sentence_index"].astype(str)
            ),
            "meeting_date": meeting_date.loc[selected.index].dt.strftime("%Y-%m-%d"),
            "period": selected["period"].astype(str),
            "committee": selected["committee"].astype(str),
            "agenda": selected["agenda"].astype(str),
            "speaker": selected["speaker"].astype(str),
            "member_id": selected["member_id"].astype(str),
            "issue_name": selected["issue_name"].astype(str),
            "issue_weight": selected["issue_weight"].astype(float),
            "matched_term_count": 1,
            "text_length": _text_length(selected),
        }
    )[OUTPUT_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"saved {len(out)} point-in-time 2025 official match rows "
        f"({out['meeting_date'].min()}..{out['meeting_date'].max()}): {out_path}",
        flush=True,
    )
    return out


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


def combine_with_15th(
    matches_15: Path,
    matches_16_22: Path,
    out_path: Path,
    matches_2025: Path | None = None,
) -> None:
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
    if matches_2025 is not None and matches_2025.exists():
        frame2025 = _read_csv(matches_2025)
        for column in OUTPUT_COLUMNS:
            if column not in frame2025.columns:
                frame2025[column] = ""
        frames.append(frame2025[OUTPUT_COLUMNS])
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
    parser.add_argument("--pres-2025-supplement", type=Path, default=DEFAULT_2025_SUPPLEMENT)
    parser.add_argument("--pres-2025-out", type=Path, default=DEFAULT_2025_OUT)
    parser.add_argument("--skip-pres-2025-supplement", action="store_true")
    parser.add_argument("--max-workbooks", type=int, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--window-mode", choices=["continuous", "campaign"], default="campaign")
    parser.add_argument(
        "--assemblies",
        nargs="+",
        default=None,
        help="Optional Assembly terms to parse, for example: --assemblies 22",
    )
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
            assemblies=set(args.assemblies) if args.assemblies else None,
        )
    if not args.skip_pres_2025_supplement and args.pres_2025_supplement.exists():
        convert_pres_2025_official_supplement(
            args.pres_2025_supplement,
            args.pres_2025_out,
            window_mode=args.window_mode,
        )
    combine_with_15th(
        args.matches_15,
        args.out,
        args.combined_out,
        None if args.skip_pres_2025_supplement else args.pres_2025_out,
    )


if __name__ == "__main__":
    main()
