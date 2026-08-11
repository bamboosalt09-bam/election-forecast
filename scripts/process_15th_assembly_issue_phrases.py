"""Run issue phrase extraction over the converted 15th Assembly workbook data."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from election_forecast.features.issue_matcher import match_issue_terms, match_issue_weights  # noqa: E402
from news_collector.sources.datalab import load_issue_keywords  # noqa: E402


INPUT_JSON = ROOT / "outputs" / "15th_assembly_conversion" / "15th_assembly_extracted.json"
OUTPUT_DIR = ROOT / "outputs" / "15th_assembly_conversion" / "issue_phrase_extraction"
KEYWORDS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "issue_keywords.csv"

SPEECH_CONTENT_COLUMNS = [f"발언내용{i}" for i in range(1, 8)]
BASE_COLUMNS = [
    "회의번호",
    "회의록구분",
    "대수",
    "회의구분",
    "위원회",
    "회수",
    "차수",
    "기타 정보",
    "회의일자",
    "안건",
    "발언자",
    "의원ID",
    "발언순번",
]


def parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"(\d{4})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def week_start(value: object) -> str:
    parsed = parse_date(value)
    if parsed is None:
        return ""
    stamp = pd.Timestamp(parsed)
    return (stamp - pd.Timedelta(days=int(stamp.weekday()))).date().isoformat()


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\uD800-\uDFFF\uFFFE\uFFFF]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_speech_rows() -> list[dict[str, Any]]:
    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    plenary_columns = payload["plenary_columns"]
    for index, values in enumerate(payload["plenary_rows"], start=1):
        record = dict(zip(plenary_columns, values))
        text = " ".join(clean_text(record.get(column)) for column in SPEECH_CONTENT_COLUMNS)
        rows.append(
            {
                **{column: clean_text(record.get(column)) for column in BASE_COLUMNS},
                "source_sheet": "본회의_xlsx",
                "source_file": "제15대 국회 본회의 회의록 데이터셋.xlsx",
                "source_page_start": "",
                "source_page_end": "",
                "extraction_status": "ok",
                "text_quality": "",
                "source_row_id": index,
                "text": clean_text(text),
            }
        )

    pdf_columns = payload["pdf_columns"]
    for index, values in enumerate(payload["pdf_rows"], start=1):
        record = dict(zip(pdf_columns, values))
        text = " ".join(clean_text(record.get(column)) for column in SPEECH_CONTENT_COLUMNS)
        rows.append(
            {
                **{column: clean_text(record.get(column)) for column in BASE_COLUMNS},
                "source_sheet": "PDF_추출",
                "source_file": clean_text(record.get("source_file")),
                "source_page_start": record.get("source_page_start") or "",
                "source_page_end": record.get("source_page_end") or "",
                "extraction_status": clean_text(record.get("extraction_status")) or "unknown",
                "text_quality": record.get("text_quality") or "",
                "source_row_id": index,
                "text": clean_text(text),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keyword_map = load_issue_keywords(KEYWORDS)
    speech_rows = load_speech_rows()

    match_rows: list[dict[str, Any]] = []
    issue_counter: Counter[tuple[str, str, str]] = Counter()
    speaker_counter: Counter[tuple[str, str, str]] = Counter()
    quality_counter: Counter[tuple[str, str, str]] = Counter()
    term_counter: Counter[tuple[str, str]] = Counter()
    source_counter: Counter[tuple[str, str]] = Counter()
    processed = 0
    matched_speeches = 0

    for row in speech_rows:
        processed += 1
        if processed % 1000 == 0:
            print(f"[match] {processed}/{len(speech_rows)}", flush=True)
        text = row["text"]
        if not text:
            continue
        terms_by_issue = match_issue_terms(text, keyword_map)
        weights_by_issue = match_issue_weights(text, keyword_map)
        if not terms_by_issue:
            continue
        matched_speeches += 1
        period = week_start(row["회의일자"])
        for issue_name, terms in sorted(terms_by_issue.items()):
            issue_weight = float(weights_by_issue.get(issue_name, 0.0))
            matched_terms = "|".join(sorted(set(terms)))
            match_rows.append(
                {
                    "source_sheet": row["source_sheet"],
                    "source_file": row["source_file"],
                    "source_row_id": row["source_row_id"],
                    "source_page_start": row["source_page_start"],
                    "source_page_end": row["source_page_end"],
                    "extraction_status": row["extraction_status"],
                    "text_quality": row["text_quality"],
                    "meeting_date": row["회의일자"],
                    "period": period,
                    "committee": row["위원회"],
                    "session": row["회수"],
                    "round": row["차수"],
                    "agenda": row["안건"],
                    "speaker": row["발언자"],
                    "member_id": row["의원ID"],
                    "speech_order": row["발언순번"],
                    "issue_name": issue_name,
                    "issue_weight": round(issue_weight, 4),
                    "matched_terms": matched_terms,
                    "matched_term_count": len(set(terms)),
                    "text_length": len(text),
                }
            )
            issue_counter[(period, issue_name, row["source_sheet"])] += 1
            quality_counter[(row["extraction_status"], row["source_sheet"], issue_name)] += 1
            speaker = row["발언자"] or "<unknown>"
            speaker_counter[(speaker, issue_name, row["source_sheet"])] += 1
            source_counter[(row["source_file"], issue_name)] += 1
            for term in set(terms):
                term_counter[(issue_name, term)] += 1

    issue_summary = [
        {
            "period": period,
            "issue_name": issue_name,
            "source_sheet": source_sheet,
            "matched_speech_issue_rows": count,
        }
        for (period, issue_name, source_sheet), count in sorted(issue_counter.items())
    ]
    speaker_summary = [
        {
            "speaker": speaker,
            "issue_name": issue_name,
            "source_sheet": source_sheet,
            "matched_speech_issue_rows": count,
        }
        for (speaker, issue_name, source_sheet), count in sorted(
            speaker_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    term_summary = [
        {
            "issue_name": issue_name,
            "term": term,
            "matched_rows": count,
        }
        for (issue_name, term), count in sorted(term_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    quality_summary = [
        {
            "extraction_status": extraction_status,
            "source_sheet": source_sheet,
            "issue_name": issue_name,
            "matched_speech_issue_rows": count,
        }
        for (extraction_status, source_sheet, issue_name), count in sorted(quality_counter.items())
    ]
    source_summary = [
        {
            "source_file": source_file,
            "issue_name": issue_name,
            "matched_speech_issue_rows": count,
        }
        for (source_file, issue_name), count in sorted(source_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    run_summary = [
        {"metric": "input_speech_rows", "value": len(speech_rows)},
        {"metric": "matched_speeches", "value": matched_speeches},
        {"metric": "match_issue_rows", "value": len(match_rows)},
        {"metric": "unique_issues", "value": len({row["issue_name"] for row in match_rows})},
        {"metric": "unique_terms", "value": len(term_counter)},
    ]

    write_csv(
        OUTPUT_DIR / "15th_assembly_issue_phrase_matches.csv",
        match_rows,
        [
            "source_sheet",
            "source_file",
            "source_row_id",
            "source_page_start",
            "source_page_end",
            "extraction_status",
            "text_quality",
            "meeting_date",
            "period",
            "committee",
            "session",
            "round",
            "agenda",
            "speaker",
            "member_id",
            "speech_order",
            "issue_name",
            "issue_weight",
            "matched_terms",
            "matched_term_count",
            "text_length",
        ],
    )
    write_csv(
        OUTPUT_DIR / "15th_assembly_issue_period_summary.csv",
        issue_summary,
        ["period", "issue_name", "source_sheet", "matched_speech_issue_rows"],
    )
    write_csv(
        OUTPUT_DIR / "15th_assembly_speaker_issue_summary.csv",
        speaker_summary,
        ["speaker", "issue_name", "source_sheet", "matched_speech_issue_rows"],
    )
    write_csv(
        OUTPUT_DIR / "15th_assembly_term_summary.csv",
        term_summary,
        ["issue_name", "term", "matched_rows"],
    )
    write_csv(
        OUTPUT_DIR / "15th_assembly_quality_summary.csv",
        quality_summary,
        ["extraction_status", "source_sheet", "issue_name", "matched_speech_issue_rows"],
    )
    write_csv(
        OUTPUT_DIR / "15th_assembly_source_issue_summary.csv",
        source_summary,
        ["source_file", "issue_name", "matched_speech_issue_rows"],
    )
    write_csv(OUTPUT_DIR / "15th_assembly_run_summary.csv", run_summary, ["metric", "value"])

    print(f"saved outputs to {OUTPUT_DIR}")
    print(f"input_speech_rows={len(speech_rows)} matched_speeches={matched_speeches} match_issue_rows={len(match_rows)}")


if __name__ == "__main__":
    main()
