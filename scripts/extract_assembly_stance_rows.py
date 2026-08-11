"""Resumable sentence-level stance extraction from Assembly transcript workbooks.

This is deliberately separate from the current issue-match aggregate and from
the forecast input. Each source workbook is emitted to an independently
verified part file, then recorded in an atomically written state file. A power
loss can at most require reprocessing the current workbook; completed parts
remain valid and are never appended to in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from election_forecast.features.issue_matcher import match_issue_weights  # noqa: E402
from extract_assembly_speaker_issue_matches import (  # noqa: E402
    ELECTION_CUTOFFS,
    assembly_daesu_from_name,
    build_keyword_inputs,
    iter_rows_from_xlsx,
    iter_workbooks_from_source,
)
from news_collector.sources.assembly_records import _week, parse_meeting_date  # noqa: E402
from process_15th_assembly_issue_phrases import load_speech_rows as load_15th_speech_rows  # noqa: E402


DEFAULT_SOURCE = ROOT / "trash_dataset"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "assembly_stance"
DEFAULT_ALIAS_PATH = ROOT / "data" / "raw" / "assembly_stance_entity_aliases.csv"
FIFTEENTH_EXTRACTED = ROOT / "outputs" / "15th_assembly_conversion" / "15th_assembly_extracted.json"
STATE_FILENAME = "state.json"
LOCK_FILENAME = ".extract.lock"
PARTS_DIRNAME = "parts"
FINAL_FILENAME = "assembly_stance_rows_15_22.csv"
LOG_FILENAME = "extract.log"
SCHEMA_VERSION = 1

DATE_COL = "회의일자"
COMMITTEE_COL = "위원회"
AGENDA_COL = "안건"
SPEAKER_COL = "발언자"
MEMBER_ID_COL = "의원ID"
SPEECH_COLS = [f"발언내용{i}" for i in range(1, 8)]

OUTPUT_COLUMNS = [
    "schema_version",
    "source_id",
    "source_file",
    "source_sha256",
    "assembly_daesu",
    "source_row_id",
    "meeting_date",
    "period",
    "available_date",
    "availability_basis",
    "election_id",
    "committee",
    "agenda",
    "speaker",
    "member_id",
    "sentence_index",
    "issue_name",
    "issue_weight",
    "target_type",
    "target_name",
    "target_alias",
    "target_model_eligible",
    "stance_label",
    "stance_polarity",
    "stance_confidence",
    "stance_cues",
    "text_excerpt",
    "text_sha256",
]

STANCE_PATTERNS = {
    "rebuttal": ("사실이 아니", "근거 없는", "근거없", "왜곡이다", "허위다", "반박한다"),
    "defend": ("변호한다", "옹호한다", "정당하다", "정당하며", "문제가 없다", "책임이 없다", "잘못이 없다"),
    "endorse": ("지지한다", "지지합니다", "적극 지지", "찬성한다", "찬성합니다", "환영한다", "환영합니다", "높이 평가한다"),
    "attack": ("비판한다", "강력히 비판", "규탄한다", "사퇴해야", "퇴진해야", "무능하다", "정책실패", "잘못했다", "부정부패", "불법이다", "은폐했다"),
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[다요])\s+(?=[가-힣A-Za-z])|\n+")
SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class EntityAlias:
    entity_type: str
    canonical_name: str
    alias: str
    model_eligible: bool


@dataclass(frozen=True)
class ExtractionSummary:
    considered: int
    processed: int
    skipped: int
    errors: int
    limited: bool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extraction_fingerprint(alias_path: Path) -> str:
    """Fingerprint every local input that changes extracted semantics."""

    semantic_inputs = [
        Path(__file__),
        alias_path,
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "issue_keywords.csv",
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "campaign_issue_terms.csv",
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "mega_issue_terms.csv",
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "issue_context_rules.csv",
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "housing_price_index_sido.csv",
    ]
    digest = hashlib.sha256()
    for path in semantic_inputs:
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii") if path.exists() else b"missing")
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _write_csv_atomic(path: Path, rows: Iterable[dict[str, object]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return count, _sha256_file(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "completed": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"state file is unreadable; do not overwrite it: {path}") from exc
    if state.get("schema_version") != SCHEMA_VERSION or not isinstance(state.get("completed"), dict):
        raise RuntimeError(f"unsupported stance extraction state schema: {path}")
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_text_atomic(path, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def _log(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"{stamp}\t{message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "RunLock":
        payload = {"token": self.token, "pid": os.getpid(), "host": socket.gethostname()}
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                return self
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"lock exists but cannot be read safely: {self.path}") from exc
                same_host = existing.get("host") == socket.gethostname()
                if same_host and not _pid_alive(int(existing.get("pid") or 0)):
                    stale = self.path.with_name(f"{self.path.name}.stale.{int(time.time())}")
                    os.replace(self.path, stale)
                    continue
                raise RuntimeError(f"another stance extraction is active: {self.path}")

    def __exit__(self, *_: object) -> None:
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass


def _clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    return SPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT.split(_clean_text(text)) if len(sentence.strip()) >= 8]


def classify_stance(sentence: str) -> tuple[str, int, float, str]:
    """Return a conservative speech-act label without inferring unsupported intent."""

    lowered = sentence.replace(" ", "")
    hits = {label: [cue for cue in cues if cue.replace(" ", "") in lowered] for label, cues in STANCE_PATTERNS.items()}
    active = [label for label, cues in hits.items() if cues]
    if not active:
        return "neutral", 0, 0.20, ""
    if "rebuttal" in active:
        return "rebuttal", 0, 0.70 if len(active) == 1 else 0.45, "|".join(hits["rebuttal"])
    if "attack" in active and ("defend" in active or "endorse" in active):
        return "ambiguous", 0, 0.25, "|".join(cue for label in active for cue in hits[label])
    if "attack" in active:
        return "attack", -1, 0.65, "|".join(hits["attack"])
    if "defend" in active:
        return "defend", 1, 0.65, "|".join(hits["defend"])
    return "endorse", 1, 0.60, "|".join(hits["endorse"])


def load_aliases(path: Path) -> list[EntityAlias]:
    if not path.exists():
        raise FileNotFoundError(f"stance entity alias file not found: {path}")
    aliases: list[EntityAlias] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            alias = (row.get("alias") or "").strip()
            canonical = (row.get("canonical_name") or "").strip()
            if alias and canonical:
                aliases.append(
                    EntityAlias(
                        entity_type=(row.get("entity_type") or "unknown").strip(),
                        canonical_name=canonical,
                        alias=alias,
                        model_eligible=(row.get("model_eligible") or "").strip().lower() in {"1", "true", "yes"},
                    )
                )
    return sorted(aliases, key=lambda item: len(item.alias), reverse=True)


def resolve_target(sentence: str, aliases: list[EntityAlias]) -> EntityAlias | None:
    for alias in aliases:
        if alias.alias in sentence:
            return alias
    return None


def election_for_date(value: object) -> str:
    parsed = parse_meeting_date(value)
    if parsed is None:
        return ""
    stamp = date.fromisoformat(parsed.isoformat())
    for election_id, (start, end) in ELECTION_CUTOFFS.items():
        start_date = date.fromisoformat(str(start)[:10])
        end_date = date.fromisoformat(str(end)[:10])
        if start_date <= stamp <= end_date:
            return election_id
    return ""


def _part_path(parts_dir: Path, source_id: str) -> Path:
    return parts_dir / f"{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:20]}.csv"


def _is_complete(
    entry: dict[str, Any],
    part_path: Path,
    source_sha256: str,
    extraction_fingerprint: str,
) -> bool:
    return (
        entry.get("source_sha256") == source_sha256
        and entry.get("extraction_fingerprint") == extraction_fingerprint
        and part_path.exists()
        and entry.get("part_sha256") == _sha256_file(part_path)
    )


def rows_for_workbook(
    *,
    source_id: str,
    source_sha256: str,
    assembly_daesu: str,
    workbook_bytes: bytes,
    aliases: list[EntityAlias],
    keyword_maps: dict[str, dict[str, list[str]]],
    term_weights: dict[str, dict[tuple[str, str], float]],
    issue_boosts: dict[str, dict[str, float]],
    context_rules: dict[str, list[object]],
) -> Iterable[dict[str, object]]:
    for row_index, row in enumerate(iter_rows_from_xlsx(workbook_bytes), 1):
        meeting = parse_meeting_date(row.get(DATE_COL))
        election_id = election_for_date(row.get(DATE_COL))
        if meeting is None or not election_id:
            continue
        text = " ".join(_clean_text(row.get(column)) for column in SPEECH_COLS if _clean_text(row.get(column)))
        if not text:
            continue
        yield from _rows_for_text(
            source_id=source_id,
            source_file=source_id,
            source_sha256=source_sha256,
            assembly_daesu=assembly_daesu,
            row_index=row_index,
            meeting=meeting,
            election_id=election_id,
            text=text,
            committee=_clean_text(row.get(COMMITTEE_COL)),
            agenda=_clean_text(row.get(AGENDA_COL)),
            speaker=_clean_text(row.get(SPEAKER_COL)),
            member_id=_clean_text(row.get(MEMBER_ID_COL)),
            aliases=aliases,
            keyword_maps=keyword_maps,
            term_weights=term_weights,
            issue_boosts=issue_boosts,
            context_rules=context_rules,
        )


def _rows_for_text(
    *,
    source_id: str,
    source_file: str,
    source_sha256: str,
    assembly_daesu: str,
    row_index: int,
    meeting: date,
    election_id: str,
    text: str,
    committee: str,
    agenda: str,
    speaker: str,
    member_id: str,
    aliases: list[EntityAlias],
    keyword_maps: dict[str, dict[str, list[str]]],
    term_weights: dict[str, dict[tuple[str, str], float]],
    issue_boosts: dict[str, dict[str, float]],
    context_rules: dict[str, list[object]],
) -> Iterable[dict[str, object]]:
    for sentence_index, sentence in enumerate(split_sentences(text), 1):
        issue_weights = match_issue_weights(
            sentence,
            keyword_maps[election_id],
            term_weights=term_weights.get(election_id),
            issue_boosts=issue_boosts.get(election_id),
            context_rules=context_rules.get(election_id),
        )
        if not issue_weights:
            continue
        target = resolve_target(sentence, aliases)
        stance_label, polarity, confidence, cues = classify_stance(sentence)
        for issue_name, issue_weight in issue_weights.items():
            yield {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "source_file": source_file,
                "source_sha256": source_sha256,
                "assembly_daesu": assembly_daesu,
                "source_row_id": row_index,
                "meeting_date": meeting.isoformat(),
                "period": _week(meeting),
                # Meeting date is a traceable proxy only. These outputs are
                # not model inputs until publication timing is separately audited.
                "available_date": meeting.isoformat(),
                "availability_basis": "meeting_date_proxy_not_model_eligible",
                "election_id": election_id,
                "committee": committee,
                "agenda": agenda,
                "speaker": speaker,
                "member_id": member_id,
                "sentence_index": sentence_index,
                "issue_name": issue_name,
                "issue_weight": round(float(issue_weight), 6),
                "target_type": target.entity_type if target else "none",
                "target_name": target.canonical_name if target else "",
                "target_alias": target.alias if target else "",
                "target_model_eligible": int(target.model_eligible) if target else 0,
                "stance_label": stance_label,
                "stance_polarity": polarity,
                "stance_confidence": confidence,
                "stance_cues": cues,
                "text_excerpt": sentence[:600],
                "text_sha256": _sha256_bytes(sentence.encode("utf-8")),
            }


def rows_for_15th_json(
    *,
    source_id: str,
    source_sha256: str,
    aliases: list[EntityAlias],
    keyword_maps: dict[str, dict[str, list[str]]],
    term_weights: dict[str, dict[tuple[str, str], float]],
    issue_boosts: dict[str, dict[str, float]],
    context_rules: dict[str, list[object]],
) -> Iterable[dict[str, object]]:
    for row_index, row in enumerate(load_15th_speech_rows(), 1):
        meeting = parse_meeting_date(row.get(DATE_COL))
        election_id = election_for_date(row.get(DATE_COL))
        text = _clean_text(row.get("text"))
        if meeting is None or not election_id or not text:
            continue
        yield from _rows_for_text(
            source_id=source_id,
            source_file=_clean_text(row.get("source_file")) or source_id,
            source_sha256=source_sha256,
            assembly_daesu="15",
            row_index=int(row.get("source_row_id") or row_index),
            meeting=meeting,
            election_id=election_id,
            text=text,
            committee=_clean_text(row.get(COMMITTEE_COL)),
            agenda=_clean_text(row.get(AGENDA_COL)),
            speaker=_clean_text(row.get(SPEAKER_COL)),
            member_id=_clean_text(row.get(MEMBER_ID_COL)),
            aliases=aliases,
            keyword_maps=keyword_maps,
            term_weights=term_weights,
            issue_boosts=issue_boosts,
            context_rules=context_rules,
        )


def extract(
    source: Path,
    output_dir: Path,
    alias_path: Path,
    assemblies: set[str],
    max_workbooks: int | None,
) -> ExtractionSummary:
    if not source.exists():
        raise FileNotFoundError(f"source archive not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / PARTS_DIRNAME
    parts_dir.mkdir(exist_ok=True)
    state_path = output_dir / STATE_FILENAME
    log_path = output_dir / LOG_FILENAME
    aliases = load_aliases(alias_path)
    keyword_maps, term_weights, issue_boosts, context_rules = build_keyword_inputs()
    extraction_fingerprint = _extraction_fingerprint(alias_path)

    with RunLock(output_dir / LOCK_FILENAME):
        state = _load_state(state_path)
        completed = state["completed"]
        considered = processed = skipped = errors = 0
        limited = False

        def process_unit(
            source_id: str,
            assembly_daesu: str,
            source_sha256: str,
            rows: Iterable[dict[str, object]],
        ) -> None:
            nonlocal processed, skipped, errors
            part_path = _part_path(parts_dir, source_id)
            entry = completed.get(source_id)
            if entry and _is_complete(entry, part_path, source_sha256, extraction_fingerprint):
                skipped += 1
                print(f"[skip-complete] {source_id}", flush=True)
                return
            try:
                row_count, part_sha256 = _write_csv_atomic(part_path, rows)
                completed[source_id] = {
                    "assembly_daesu": assembly_daesu,
                    "source_sha256": source_sha256,
                    "extraction_fingerprint": extraction_fingerprint,
                    "part_file": str(part_path.relative_to(output_dir)),
                    "part_sha256": part_sha256,
                    "row_count": row_count,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                _save_state(state_path, state)
                _log(log_path, f"complete\t{source_id}\trows={row_count}\tsha256={source_sha256}")
                processed += 1
                print(f"[complete] {source_id} rows={row_count}", flush=True)
            except Exception as exc:  # noqa: BLE001 - resume must continue past a bad source
                errors += 1
                _log(log_path, f"error\t{source_id}\t{exc!r}")
                print(f"[error] {source_id}: {exc!r}", flush=True)

        if "15" in assemblies:
            considered += 1
            if max_workbooks is not None and considered > max_workbooks:
                limited = True
            elif not FIFTEENTH_EXTRACTED.exists():
                errors += 1
                _log(log_path, f"error\t15th_converted_json\tmissing={FIFTEENTH_EXTRACTED}")
                print(f"[error] missing 15th converted source: {FIFTEENTH_EXTRACTED}", flush=True)
            else:
                source_id = "15th_assembly_extracted.json"
                source_sha256 = _sha256_file(FIFTEENTH_EXTRACTED)
                process_unit(
                    source_id,
                    "15",
                    source_sha256,
                    rows_for_15th_json(
                        source_id=source_id,
                        source_sha256=source_sha256,
                        aliases=aliases,
                        keyword_maps=keyword_maps,
                        term_weights=term_weights,
                        issue_boosts=issue_boosts,
                        context_rules=context_rules,
                    ),
                )

        for source_id, workbook_bytes in iter_workbooks_from_source(source):
            assembly_daesu = assembly_daesu_from_name(source_id)
            # The converted 15th JSON already contains the 15th plenary source
            # and extracted committee text. Reading the raw 15th workbooks here
            # would duplicate those rows, so 15 is represented only by the JSON
            # source unit above.
            if assembly_daesu == "15":
                continue
            if assembly_daesu not in assemblies:
                continue
            considered += 1
            if max_workbooks is not None and considered > max_workbooks:
                limited = True
                break
            source_sha256 = _sha256_bytes(workbook_bytes)
            process_unit(
                source_id,
                assembly_daesu,
                source_sha256,
                rows_for_workbook(
                    source_id=source_id,
                    source_sha256=source_sha256,
                    assembly_daesu=assembly_daesu,
                    workbook_bytes=workbook_bytes,
                    aliases=aliases,
                    keyword_maps=keyword_maps,
                    term_weights=term_weights,
                    issue_boosts=issue_boosts,
                    context_rules=context_rules,
                ),
            )
        summary = ExtractionSummary(considered, processed, skipped, errors, limited)
        state["last_run"] = {
            "considered": considered,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "limited": limited,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if errors or limited:
            state["final_valid"] = False
        _save_state(state_path, state)
        print(
            f"[summary] considered={considered} processed={processed} skipped={skipped} "
            f"errors={errors} limited={limited} output={output_dir}",
            flush=True,
        )
        return summary


def finalize(output_dir: Path) -> Path:
    state_path = output_dir / STATE_FILENAME
    state = _load_state(state_path)
    final_path = output_dir / FINAL_FILENAME
    temp = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for source_id, entry in sorted(state["completed"].items()):
            part_path = output_dir / entry["part_file"]
            if not _is_complete(
                entry,
                part_path,
                entry["source_sha256"],
                entry.get("extraction_fingerprint", ""),
            ):
                raise RuntimeError(f"cannot finalize corrupt or incomplete part: {source_id}")
            with part_path.open("r", encoding="utf-8-sig", newline="") as part:
                for row in csv.DictReader(part):
                    writer.writerow(row)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temp, final_path)
    state["final_valid"] = True
    state["final_sha256"] = _sha256_file(final_path)
    state["finalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _save_state(state_path, state)
    print(f"[finalized] {final_path}", flush=True)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--assemblies", default="15,16,17,18,19,20,21,22")
    parser.add_argument("--max-workbooks", type=int, default=None)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assemblies = {item.strip() for item in args.assemblies.split(",") if item.strip()}
    summary = extract(args.source, args.output_dir, args.aliases, assemblies, args.max_workbooks)
    if args.finalize:
        if summary.errors or summary.limited:
            raise SystemExit("refusing to finalize: resume until every requested source completes without errors")
        finalize(args.output_dir)


if __name__ == "__main__":
    main()
