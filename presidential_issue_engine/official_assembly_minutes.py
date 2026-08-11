"""Parsers for the official National Assembly minutes website.

The website exposes a hierarchy through small HTML fragments rather than a
documented JSON API.  These helpers keep HTML interpretation separate from the
network collector so fixtures can verify every discovery step offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

from bs4 import BeautifulSoup


BASE_URL = "https://record.assembly.go.kr"
ASSEMBLY_TERM = "22"
CLASS_NAMES = {
    "1": "plenary",
    "2": "standing_committee",
    "3": "special_committee",
    "4": "budget_committee",
    "5": "inspection",
    "6": "investigation",
}

_DATE_RE = re.compile(r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")
_PDF_DATE_RE = re.compile(rb"/CreationDate\s*\(D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")


@dataclass(frozen=True)
class Committee:
    class_id: str
    committee_code: str
    committee_name: str


@dataclass(frozen=True)
class Meeting:
    minutes_id: str
    class_id: str
    class_name: str
    committee_code: str
    committee_name: str
    session: str
    meeting_date: date
    title: str

    @property
    def viewer_url(self) -> str:
        return f"{BASE_URL}/assembly/viewer/minutes/xml.do?id={self.minutes_id}&type=view"

    @property
    def summary_url(self) -> str:
        return f"{BASE_URL}/assembly/viewer/minutes/xml.do?id={self.minutes_id}&type=summary"

    @property
    def pdf_url(self) -> str:
        return f"{BASE_URL}/assembly/viewer/minutes/download/pdf.do?id={self.minutes_id}"


@dataclass(frozen=True)
class SpeakerBlock:
    block_index: int
    speaker_name: str
    member_id: str
    position: str
    text: str


def _text(node: object) -> str:
    get_text = getattr(node, "get_text", None)
    return get_text(" ", strip=True) if get_text else ""


def parse_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    try:
        return date(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def parse_sessions(html: str, class_id: str) -> set[str]:
    """Return every session advertised by a hierarchy fragment."""

    soup = BeautifulSoup(html, "html.parser")
    return {
        str(node.get("data-sess") or "").strip()
        for node in soup.select(f'[data-class="{class_id}"][data-sess]')
        if str(node.get("data-sess") or "").strip()
    }


def parse_session_windows(html: str, class_id: str = "1") -> dict[str, tuple[date, date]]:
    """Read advertised session ranges from the term hierarchy page."""

    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, tuple[date, date]] = {}
    for node in soup.select(f'[data-class="{class_id}"][data-sess]'):
        session = str(node.get("data-sess") or "").strip()
        dates = []
        for match in _DATE_RE.finditer(_text(node)):
            try:
                dates.append(date(*(int(value) for value in match.groups())))
            except ValueError:
                continue
        if session and len(dates) >= 2:
            result[session] = (dates[0], dates[1])
    return result


def parse_years(html: str, class_id: str = "5") -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    return {
        str(node.get("data-dt") or "").strip()
        for node in soup.select(f'[data-class="{class_id}"][data-dt]')
        if str(node.get("data-dt") or "").strip()
    }


def parse_committees(html: str, class_id: str) -> list[Committee]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    result: list[Committee] = []
    for node in soup.select(f'[data-class="{class_id}"][data-cmit]'):
        code = str(node.get("data-cmit") or "").strip()
        name = _text(node)
        if not code or not name or code in seen:
            continue
        seen.add(code)
        result.append(Committee(class_id, code, name))
    return result


def parse_meetings(
    html: str,
    *,
    class_id: str,
    committee_code: str = "",
    committee_name: str = "",
    session: str = "",
) -> list[Meeting]:
    """Parse meeting IDs and dates from a final hierarchy fragment."""

    soup = BeautifulSoup(html, "html.parser")
    result: list[Meeting] = []
    seen: set[str] = set()
    for node in soup.select("[data-id]"):
        minutes_id = str(node.get("data-id") or "").strip()
        if not minutes_id.isdigit() or minutes_id in seen:
            continue
        title = str(node.get("title") or "").strip() or _text(node)
        meeting_date = parse_date(title)
        if meeting_date is None:
            continue
        seen.add(minutes_id)
        result.append(
            Meeting(
                minutes_id=minutes_id,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                committee_code=committee_code,
                committee_name=committee_name or CLASS_NAMES[class_id],
                session=str(node.get("data-sess") or session or "").strip(),
                meeting_date=meeting_date,
                title=title,
            )
        )
    return result


def parse_pdf_creation_datetime(pdf_bytes: bytes) -> datetime | None:
    """Read the official PDF CreationDate without a PDF runtime dependency."""

    match = _PDF_DATE_RE.search(pdf_bytes)
    if not match:
        return None
    try:
        return datetime(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def conservative_available_date(
    pdf_bytes: bytes,
    *,
    collected_on: date,
    safety_lag_days: int = 1,
) -> tuple[date, str, str]:
    """Return a fail-closed day-level availability proxy.

    The site does not expose an exact publication timestamp.  A final PDF
    cannot have been public before its embedded creation time, so one full-day
    quarantine is added.  If metadata is absent, the collection date is the
    only verified upper bound and therefore makes historical use ineligible.
    """

    created = parse_pdf_creation_datetime(pdf_bytes)
    if created is None:
        return collected_on, "collection_date_fallback_not_historical", ""
    available = created.date() + timedelta(days=max(0, safety_lag_days))
    return available, "official_pdf_creation_plus_safety_lag", created.isoformat()


def parse_viewer_header(html: str) -> tuple[str, date | None]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.select_one("h2") or soup.select_one("h1")
    title = _text(heading)
    return title, parse_date(title)


def parse_speaker_blocks(html: str) -> list[SpeakerBlock]:
    """Extract only attributed speech, excluding navigation and captions."""

    soup = BeautifulSoup(html, "html.parser")
    result: list[SpeakerBlock] = []
    for index, node in enumerate(soup.select(".speaker"), 1):
        utterances = [_text(part) for part in node.select(".spk_sub")]
        text = " ".join(value for value in utterances if value).strip()
        if not text:
            continue
        name = str(node.get("data-name") or "").strip()
        if not name:
            name_node = node.select_one(".name")
            name = _text(name_node)
        result.append(
            SpeakerBlock(
                block_index=index,
                speaker_name=name,
                member_id=str(node.get("data-mem_id") or "").strip(),
                position=str(node.get("data-pos") or "").strip(),
                text=text,
            )
        )
    return result


def meetings_in_window(
    meetings: Iterable[Meeting], start: date, end: date
) -> list[Meeting]:
    return [meeting for meeting in meetings if start <= meeting.meeting_date <= end]
