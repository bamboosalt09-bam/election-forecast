"""Collect all official Assembly minutes relevant to the 2025 demo cutoff.

Raw HTML/PDF files and per-meeting parts are cached outside Git tracking.  The
collector is idempotent and atomically checkpoints after every meeting.  It
stores every meeting held from 2025-01-01 through 2025-06-02, but model
eligibility is based on the official PDF creation date plus a one-day safety
lag rather than the meeting date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Iterable

import httpx
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from news_collector.sources.datalab import load_issue_keywords  # noqa: E402
from presidential_issue_engine.official_assembly_minutes import (  # noqa: E402
    ASSEMBLY_TERM,
    BASE_URL,
    CLASS_NAMES,
    Meeting,
    conservative_available_date,
    meetings_in_window,
    parse_committees,
    parse_meetings,
    parse_session_windows,
    parse_sessions,
    parse_speaker_blocks,
    parse_viewer_header,
    parse_years,
)
from extract_assembly_stance_rows import (  # noqa: E402
    DEFAULT_ALIAS_PATH,
    OUTPUT_COLUMNS,
    _rows_for_text,
    load_aliases,
)


START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 6, 2)
TARGET_ELECTION = "pres_2025"
SAFETY_LAG_DAYS = 1
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/raw/official_sources/assembly_pres_2025_minutes"
)
DEFAULT_CACHE_DIR = (
    ROOT / "data/raw/official_sources/cache/assembly_pres_2025_minutes"
)
DEFAULT_CHECKPOINT = (
    ROOT / "data/raw/official_sources/checkpoints/assembly_pres_2025_minutes.json"
)
KEYWORDS = ROOT / "presidential_issue_engine/fixed_dataset/issue_keywords.csv"
FINAL_FILENAME = "assembly_stance_rows_2025_h1.csv"
MEETING_MANIFEST_FILENAME = "meeting_manifest.csv"
COLLECTION_MANIFEST_FILENAME = "manifest.json"
FORBIDDEN_OUTCOME_COLUMNS = {
    "actual_vote_share",
    "candidate_votes",
    "error",
    "mae",
    "mean_vote_share",
    "pred",
    "vote_share",
    "votes",
    "winner",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _atomic_csv(
    path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, _sha256_file(path)


def _semantic_fingerprint(alias_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        ROOT / "presidential_issue_engine/official_assembly_minutes.py",
        ROOT / "scripts/extract_assembly_stance_rows.py",
        KEYWORDS,
        alias_path,
    ):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


class OfficialMinutesClient:
    def __init__(self, delay_seconds: float = 0.10) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self.client = httpx.Client(
            timeout=httpx.Timeout(90.0, connect=30.0),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; election-forecast-research/1.0)",
                "Referer": f"{BASE_URL}/assembly/mnts/total/{ASSEMBLY_TERM}.do",
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OfficialMinutesClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def request(self, method: str, path: str, **kwargs: object) -> bytes:
        url = path if path.startswith("http") else f"{BASE_URL}{path}"
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                headers = dict(kwargs.pop("headers", {}) or {})
                if method.upper() == "POST":
                    headers["X-Requested-With"] = "XMLHttpRequest"
                response = self.client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                if self.delay_seconds:
                    time.sleep(self.delay_seconds)
                return response.content
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if 400 <= status < 500 and status not in {408, 429}:
                    raise RuntimeError(
                        f"official minutes request rejected: {status} {method} {url}"
                    ) from exc
                last_error = exc
                time.sleep(min(2**attempt, 16))
            except (httpx.TransportError, OSError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"official minutes request failed: {method} {url}") from last_error

    def text(self, method: str, path: str, **kwargs: object) -> str:
        return self.request(method, path, **kwargs).decode("utf-8", "replace")


def _post_data(**overrides: str) -> dict[str, str]:
    data = {
        "th_sch": ASSEMBLY_TERM,
        "class_id_sch": "",
        "sess_sch": "",
        "chk": "all",
        "cmit_id_sch": "",
        "cmit_cd_sch": "",
        "cmit_chk": "all",
        "cmit_chk_cmit": "",
        "cmit_chk_sub": "",
        "cmit_chk_etc": "",
        "mnts_id_sch": "",
        "mnts_year_sch": "",
        "conf_year": "",
        "council_cd_sch": "",
    }
    data.update(overrides)
    return data


def discover_meetings(client: OfficialMinutesClient) -> list[Meeting]:
    """Walk all six official meeting classes and return a complete date slice."""

    plenary_page = client.text(
        "GET",
        f"/assembly/mnts/total/{ASSEMBLY_TERM}.do",
        params={"class_id_sch": "1", "cmit_chk": "all"},
    )
    session_windows = parse_session_windows(plenary_page)
    relevant_sessions = {
        session
        for session, (start, end) in session_windows.items()
        if start <= END_DATE and end >= START_DATE
    }
    if not relevant_sessions:
        raise RuntimeError("official hierarchy exposed no sessions for the target window")

    discovered: dict[str, Meeting] = {}
    for class_id in CLASS_NAMES:
        main_page = client.text(
            "GET",
            f"/assembly/mnts/total/{ASSEMBLY_TERM}.do",
            params={"class_id_sch": class_id, "cmit_chk": "all"},
        )
        if class_id in {"1", "4"}:
            for session in sorted(parse_sessions(main_page, class_id) & relevant_sessions):
                fragment = client.text(
                    "POST",
                    "/assembly/mnts/async/sess.do",
                    data=_post_data(class_id_sch=class_id, sess_sch=session),
                )
                for meeting in meetings_in_window(
                    parse_meetings(fragment, class_id=class_id, session=session),
                    START_DATE,
                    END_DATE,
                ):
                    discovered[meeting.minutes_id] = meeting
            continue

        committees = parse_committees(main_page, class_id)
        if not committees:
            raise RuntimeError(f"official hierarchy exposed no committees for class {class_id}")
        for committee in committees:
            hierarchy = client.text(
                "POST",
                "/assembly/mnts/async/sessCmit.do",
                data=_post_data(
                    class_id_sch=class_id,
                    cmit_id_sch=committee.committee_code,
                ),
            )
            if class_id == "5":
                for year in sorted(parse_years(hierarchy)):
                    if year != "2025":
                        continue
                    fragment = client.text(
                        "POST",
                        "/assembly/mnts/async/cmit.do",
                        data=_post_data(
                            class_id_sch=class_id,
                            cmit_cd_sch=committee.committee_code,
                            conf_year=year,
                        ),
                    )
                    meetings = parse_meetings(
                        fragment,
                        class_id=class_id,
                        committee_code=committee.committee_code,
                        committee_name=committee.committee_name,
                    )
                    for meeting in meetings_in_window(meetings, START_DATE, END_DATE):
                        discovered[meeting.minutes_id] = meeting
                continue

            for session in sorted(parse_sessions(hierarchy, class_id) & relevant_sessions):
                fragment = client.text(
                    "POST",
                    "/assembly/mnts/async/cmit.do",
                    data=_post_data(
                        class_id_sch=class_id,
                        sess_sch=session,
                        cmit_cd_sch=committee.committee_code,
                    ),
                )
                meetings = parse_meetings(
                    fragment,
                    class_id=class_id,
                    committee_code=committee.committee_code,
                    committee_name=committee.committee_name,
                    session=session,
                )
                for meeting in meetings_in_window(meetings, START_DATE, END_DATE):
                    discovered[meeting.minutes_id] = meeting

    meetings = sorted(
        discovered.values(), key=lambda item: (item.meeting_date, int(item.minutes_id))
    )
    if not meetings:
        raise RuntimeError("official hierarchy discovery returned zero target meetings")
    return meetings


def _load_state(path: Path, fingerprint: str) -> dict[str, object]:
    if not path.exists():
        return {"schema": "official_assembly_pres_2025_checkpoint_v1", "completed": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state.get("completed"), dict):
        raise RuntimeError("invalid official-minutes checkpoint")
    if state.get("semantic_fingerprint") not in {None, fingerprint}:
        # Raw cache stays reusable, but every semantic part must be regenerated.
        state["completed"] = {}
    return state


def _cached_or_fetch(
    client: OfficialMinutesClient,
    path: Path,
    url: str,
) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    payload = client.request("GET", url)
    _atomic_bytes(path, payload)
    return payload


def _validated_viewer_html(
    client: OfficialMinutesClient,
    cache_dir: Path,
    meeting: Meeting,
) -> bytes | None:
    """Return viewer HTML only after its embedded meeting date is verified.

    The official server can transiently return a different meeting for a valid
    ID. Invalid payloads are retained for audit, then a no-cache request is
    retried. An unverified page is never allowed into an extracted part.
    """

    path = cache_dir / "html" / f"{meeting.minutes_id}.html"
    candidates: list[bytes] = []
    if path.exists() and path.stat().st_size > 0:
        candidates.append(path.read_bytes())
    for attempt in range(4):
        if candidates:
            payload = candidates.pop(0)
        else:
            try:
                payload = client.request(
                    "GET",
                    meeting.viewer_url,
                    headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
                )
            except RuntimeError:
                # Some official records expose only HWP/PDF downloads and
                # consistently reject the XML viewer. PDF text is the complete
                # failover source for those records.
                return None
        _, observed_date = parse_viewer_header(payload.decode("utf-8", "replace"))
        if observed_date == meeting.meeting_date:
            _atomic_bytes(path, payload)
            return payload
        invalid_path = (
            cache_dir
            / "invalid_html"
            / f"{meeting.minutes_id}-{_sha256_bytes(payload)[:16]}.html"
        )
        if not invalid_path.exists():
            _atomic_bytes(invalid_path, payload)
    raise RuntimeError(
        f"official viewer repeatedly returned the wrong meeting: {meeting.minutes_id}"
    )


def _part_is_complete(entry: dict[str, object], part_path: Path, fingerprint: str) -> bool:
    return (
        entry.get("semantic_fingerprint") == fingerprint
        and part_path.exists()
        and entry.get("part_sha256") == _sha256_file(part_path)
    )


def _meeting_rows(
    meeting: Meeting,
    viewer_html: bytes | None,
    pdf_bytes: bytes,
    *,
    aliases: list[object],
    collected_on: date,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    html = viewer_html.decode("utf-8", "replace") if viewer_html else ""
    viewer_title, viewer_date = parse_viewer_header(html) if html else (meeting.title, None)
    if html and viewer_date != meeting.meeting_date:
        raise RuntimeError(
            f"viewer date mismatch for {meeting.minutes_id}: {viewer_date} != {meeting.meeting_date}"
        )
    available, availability_basis, pdf_created = conservative_available_date(
        pdf_bytes,
        collected_on=collected_on,
        safety_lag_days=SAFETY_LAG_DAYS,
    )
    source_sha = _sha256_bytes(viewer_html) if viewer_html else _sha256_bytes(pdf_bytes)
    keyword_maps = {TARGET_ELECTION: load_issue_keywords(KEYWORDS)}
    term_weights = {TARGET_ELECTION: {}}
    issue_boosts = {TARGET_ELECTION: {}}
    context_rules = {TARGET_ELECTION: []}
    rows: list[dict[str, object]] = []
    blocks = parse_speaker_blocks(html) if html else []
    source_blocks: list[tuple[int, str, str, str]]
    if blocks:
        source_mode = "official_xml_attributed_speech"
        source_blocks = [
            (block.block_index, block.text, block.speaker_name, block.member_id)
            for block in blocks
        ]
    else:
        source_mode = "official_pdf_page_text_fallback"
        reader = PdfReader(io.BytesIO(pdf_bytes))
        source_blocks = [
            (page_index, text, "", "")
            for page_index, page in enumerate(reader.pages, 1)
            if (text := (page.extract_text() or "").strip())
        ]
        availability_basis = f"{availability_basis}_pdf_text_fallback"
    for block_index, block_text, speaker_name, member_id in source_blocks:
        extracted = _rows_for_text(
            source_id=f"official_minutes_{meeting.minutes_id}",
            source_file=meeting.viewer_url if html else meeting.pdf_url,
            source_sha256=source_sha,
            assembly_daesu=ASSEMBLY_TERM,
            row_index=block_index,
            meeting=meeting.meeting_date,
            election_id=TARGET_ELECTION,
            text=block_text,
            committee=meeting.committee_name,
            agenda=viewer_title or meeting.title,
            speaker=speaker_name,
            member_id=member_id,
            aliases=aliases,
            keyword_maps=keyword_maps,
            term_weights=term_weights,
            issue_boosts=issue_boosts,
            context_rules=context_rules,
        )
        for row in extracted:
            row["available_date"] = available.isoformat()
            row["availability_basis"] = availability_basis
            rows.append(row)
    return rows, {
        "viewer_title": viewer_title,
        "speaker_blocks": len(blocks),
        "source_blocks": len(source_blocks),
        "text_source_mode": source_mode,
        "issue_rows": len(rows),
        "pdf_creation_datetime": pdf_created,
        "available_date": available.isoformat(),
        "availability_basis": availability_basis,
        "model_eligible_at_cutoff": available <= END_DATE,
    }


def collect(
    *,
    output_dir: Path,
    cache_dir: Path,
    checkpoint: Path,
    alias_path: Path,
    delay_seconds: float,
    discover_only: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _semantic_fingerprint(alias_path)
    state = _load_state(checkpoint, fingerprint)
    state["semantic_fingerprint"] = fingerprint
    state["target_start"] = START_DATE.isoformat()
    state["target_end"] = END_DATE.isoformat()
    collected_on = date.today()
    aliases = load_aliases(alias_path)

    with OfficialMinutesClient(delay_seconds) as client:
        meetings = discover_meetings(client)
        discovery_rows = [
            {
                "minutes_id": meeting.minutes_id,
                "class_id": meeting.class_id,
                "class_name": meeting.class_name,
                "committee_code": meeting.committee_code,
                "committee_name": meeting.committee_name,
                "session": meeting.session,
                "meeting_date": meeting.meeting_date.isoformat(),
                "title": meeting.title,
                "viewer_url": meeting.viewer_url,
                "pdf_url": meeting.pdf_url,
            }
            for meeting in meetings
        ]
        discovery_path = output_dir / "discovered_meetings.csv"
        _atomic_csv(discovery_path, discovery_rows, list(discovery_rows[0]))
        state["discovered_meeting_ids"] = [meeting.minutes_id for meeting in meetings]
        state["discovered_meeting_count"] = len(meetings)
        _atomic_json(checkpoint, state)
        print(f"[discovered] meetings={len(meetings)}", flush=True)
        if discover_only:
            return {"status": "discovered_only", "meetings": len(meetings)}

        completed = state["completed"]
        assert isinstance(completed, dict)
        for index, meeting in enumerate(meetings, 1):
            part_path = cache_dir / "parts" / f"{meeting.minutes_id}.csv"
            existing = completed.get(meeting.minutes_id)
            if isinstance(existing, dict) and _part_is_complete(
                existing, part_path, fingerprint
            ):
                print(
                    f"[skip {index}/{len(meetings)}] {meeting.minutes_id} {meeting.meeting_date}",
                    flush=True,
                )
                continue
            pdf_path = cache_dir / "pdf" / f"{meeting.minutes_id}.pdf"
            pdf_bytes = _cached_or_fetch(client, pdf_path, meeting.pdf_url)
            viewer_html = _validated_viewer_html(client, cache_dir, meeting)
            rows, metadata = _meeting_rows(
                meeting,
                viewer_html,
                pdf_bytes,
                aliases=aliases,
                collected_on=collected_on,
            )
            row_count, part_sha = _atomic_csv(part_path, rows, OUTPUT_COLUMNS)
            completed[meeting.minutes_id] = {
                **metadata,
                "minutes_id": meeting.minutes_id,
                "meeting_date": meeting.meeting_date.isoformat(),
                "class_id": meeting.class_id,
                "class_name": meeting.class_name,
                "committee_code": meeting.committee_code,
                "committee_name": meeting.committee_name,
                "session": meeting.session,
                "title": meeting.title,
                "viewer_url": meeting.viewer_url,
                "pdf_url": meeting.pdf_url,
                "viewer_sha256": _sha256_bytes(viewer_html) if viewer_html else "",
                "pdf_sha256": _sha256_bytes(pdf_bytes),
                "part_path": str(part_path),
                "part_sha256": part_sha,
                "issue_rows": row_count,
                "semantic_fingerprint": fingerprint,
            }
            state["completed"] = completed
            state["last_completed_id"] = meeting.minutes_id
            _atomic_json(checkpoint, state)
            print(
                f"[complete {index}/{len(meetings)}] {meeting.minutes_id} "
                f"date={meeting.meeting_date} available={metadata['available_date']} rows={row_count}",
                flush=True,
            )

    missing = [
        meeting.minutes_id for meeting in meetings if meeting.minutes_id not in completed
    ]
    if missing:
        raise RuntimeError(f"collector ended with incomplete meetings: {missing[:10]}")

    manifest_rows: list[dict[str, object]] = []
    for meeting in meetings:
        entry = completed[meeting.minutes_id]
        assert isinstance(entry, dict)
        manifest_rows.append({key: value for key, value in entry.items() if key != "part_path"})
    meeting_manifest_path = output_dir / MEETING_MANIFEST_FILENAME
    meeting_fields = list(manifest_rows[0])
    _atomic_csv(meeting_manifest_path, manifest_rows, meeting_fields)

    final_path = output_dir / FINAL_FILENAME

    def final_rows() -> Iterable[dict[str, object]]:
        seen: set[tuple[str, str, str, str]] = set()
        for meeting in meetings:
            entry = completed[meeting.minutes_id]
            assert isinstance(entry, dict)
            part_path = Path(str(entry["part_path"]))
            if not _part_is_complete(entry, part_path, fingerprint):
                raise RuntimeError(f"corrupt meeting part: {meeting.minutes_id}")
            with part_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    forbidden = set(row) & FORBIDDEN_OUTCOME_COLUMNS
                    if forbidden:
                        raise RuntimeError(f"outcome columns in meeting part: {sorted(forbidden)}")
                    key = (
                        row["source_id"],
                        row["source_row_id"],
                        row["sentence_index"],
                        row["issue_name"],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    yield row

    final_count, final_sha = _atomic_csv(final_path, final_rows(), OUTPUT_COLUMNS)
    eligible_meetings = sum(
        bool(completed[item.minutes_id]["model_eligible_at_cutoff"]) for item in meetings
    )
    collection_manifest = {
        "schema": "official_assembly_pres_2025_collection_v1",
        "status": "forecast_only_not_scored",
        "target_election": TARGET_ELECTION,
        "meeting_window_start": START_DATE.isoformat(),
        "meeting_window_end": END_DATE.isoformat(),
        "availability_policy": "official PDF CreationDate plus one full day",
        "availability_policy_caveat": (
            "The official site exposes no exact publication timestamp; PDF creation plus "
            "one day is a conservative proxy, and missing metadata fails closed."
        ),
        "safety_lag_days": SAFETY_LAG_DAYS,
        "meetings_discovered": len(meetings),
        "meetings_completed": len(completed),
        "meetings_eligible_at_cutoff": eligible_meetings,
        "meetings_excluded_by_availability": len(meetings) - eligible_meetings,
        "first_meeting_date": meetings[0].meeting_date.isoformat(),
        "last_meeting_date": meetings[-1].meeting_date.isoformat(),
        "semantic_fingerprint": fingerprint,
        "pres_2025_outcome_used": False,
        "performance_metrics_computed": False,
        "outcome_columns_read": [],
        "outputs": {
            "discovered_meetings.csv": {
                "rows": len(meetings),
                "sha256": _sha256_file(output_dir / "discovered_meetings.csv"),
            },
            MEETING_MANIFEST_FILENAME: {
                "rows": len(manifest_rows),
                "sha256": _sha256_file(meeting_manifest_path),
            },
            FINAL_FILENAME: {"rows": final_count, "sha256": final_sha},
        },
    }
    _atomic_json(output_dir / COLLECTION_MANIFEST_FILENAME, collection_manifest)
    state["final_sha256"] = final_sha
    state["final_rows"] = final_count
    state["final_valid"] = True
    _atomic_json(checkpoint, state)
    print(
        f"[finalized] meetings={len(meetings)} eligible={eligible_meetings} "
        f"rows={final_count} output={final_path}",
        flush=True,
    )
    return collection_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--delay-seconds", type=float, default=0.10)
    parser.add_argument("--discover-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        alias_path=args.aliases,
        delay_seconds=args.delay_seconds,
        discover_only=args.discover_only,
    )


if __name__ == "__main__":
    main()
