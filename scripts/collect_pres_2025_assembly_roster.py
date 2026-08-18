"""Collect the 22nd-Assembly roster entries needed by the 2025 speech slice.

Only speakers already present in the D-1-bounded minutes input are queried.
The collector keeps the party and constituency aligned with the ``22nd`` term
entry and discards biographies and other fields that can change after cutoff.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from news_collector.sources.member_party import party_bloc  # noqa: E402


API_URL = "https://open.assembly.go.kr/portal/openapi/ALLNAMEMBER"
TARGET_TERM = "\uc81c22\ub300"
AVAILABLE_DATE = "2024-04-11"
DEFAULT_SPEAKERS = (
    ROOT
    / "data/raw/official_sources/assembly_pres_2025_context"
    / "pres_2025_speaker_issue_matches.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/raw/official_sources/assembly_pres_2025_context"
    / "assembly22_speaker_roster.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _term_index(record: dict[str, object], term: str = TARGET_TERM) -> int | None:
    terms = [value.strip() for value in str(record.get("GTELT_ERACO") or "").split(",")]
    return terms.index(term) if term in terms else None


def _aligned_term_value(value: object, index: int) -> str:
    parts = [part.strip() for part in str(value or "").split("/")]
    if not parts or parts == [""]:
        return ""
    return parts[index] if index < len(parts) else parts[-1]


def _query_name(name: str) -> list[dict[str, object]]:
    query = urlencode({"Type": "json", "NAAS_NM": name})
    request = Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": "election-forecast-roster-collector/1.0"},
    )
    try:
        with urlopen(request, timeout=45) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError):
        return []
    blocks = payload.get("ALLNAMEMBER", [])
    if len(blocks) < 2:
        return []
    return list(blocks[1].get("row", []))


def _select_record(name: str, records: list[dict[str, object]]) -> dict[str, object] | None:
    candidates = []
    for record in records:
        index = _term_index(record)
        if str(record.get("NAAS_NM") or "").strip() == name and index is not None:
            candidates.append((record, index))
    if len(candidates) != 1:
        return None
    record, index = candidates[0]
    party = _aligned_term_value(record.get("PLPT_NM"), index)
    district = _aligned_term_value(record.get("ELECD_NM"), index)
    mandate = _aligned_term_value(record.get("ELECD_DIV_NM"), index)
    if not party:
        return None
    return {
        "daesu": 22,
        "name": name,
        "party": party,
        "bloc": party_bloc(party),
        "district": district,
        "mandate_label": mandate,
        "source_member_code": str(record.get("NAAS_CD") or ""),
        "available_date": AVAILABLE_DATE,
        "source_url": API_URL,
        "source_type": "official_assembly_member_term_registry",
    }


def collect(speakers_path: Path, output: Path, *, pause_seconds: float = 0.03) -> dict[str, object]:
    speakers = pd.read_csv(speakers_path, encoding="utf-8-sig", usecols=["speaker"])
    names = sorted(
        name
        for name in speakers["speaker"].dropna().astype(str).str.strip().unique()
        if name
    )
    rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for position, name in enumerate(names, start=1):
        selected = _select_record(name, _query_name(name))
        if selected is None:
            unmatched.append(name)
        else:
            rows.append(selected)
        if position % 25 == 0 or position == len(names):
            print(
                f"queried {position}/{len(names)} speakers; matched {len(rows)}",
                flush=True,
            )
        if pause_seconds > 0.0:
            time.sleep(pause_seconds)

    roster = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        roster.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    )
    manifest = {
        "schema": "assembly22_speaker_roster_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_term": TARGET_TERM,
        "party_value_policy": "term-index-aligned PLPT_NM value",
        "available_date": AVAILABLE_DATE,
        "source_url": API_URL,
        "speaker_input": speakers_path.relative_to(ROOT).as_posix(),
        "speaker_input_sha256": _sha256(speakers_path),
        "queried_names": len(names),
        "matched_members": len(roster),
        "unmatched_names": unmatched,
        "discarded_api_fields": "all fields except term-aligned name, party, district, mandate and member code",
        "target_election_outcomes_used": False,
        "output_sha256": _sha256(output),
    }
    output.with_suffix(".manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pause-seconds", type=float, default=0.03)
    args = parser.parse_args()
    manifest = collect(args.speakers.resolve(), args.output.resolve(), pause_seconds=args.pause_seconds)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
