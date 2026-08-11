"""Collect official presidential candidate rosters from 1997 through 2022."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from news_collector.sources.nec_api import (  # noqa: E402
    CANDIDATE_REGISTRY_BASE,
    fetch_registered_candidates,
)


ELECTIONS = {
    "pres_1997": "19971218",
    "pres_2002": "20021219",
    "pres_2007": "20071219",
    "pres_2012": "20121219",
    "pres_2017": "20170509",
    "pres_2022": "20220309",
}
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "official_sources"


def _clean(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    cache_dir = args.output_dir / "cache" / "nec_presidential_candidate_registry"
    collected_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    page_provenance: list[dict[str, Any]] = []
    for election_id, sg_id in ELECTIONS.items():
        records, provenance = fetch_registered_candidates(
            sg_id=sg_id,
            sg_typecode="1",
            cache_dir=cache_dir,
            offline=args.offline,
            refresh=args.refresh,
        )
        page_provenance.extend(
            [{"election_id": election_id, **item} for item in provenance]
        )
        for record in records:
            registration_date = _clean(record.get("regdate"))
            if len(registration_date) == 8 and registration_date.isdigit():
                registration_available_date = pd.Timestamp(
                    registration_date
                ).date().isoformat()
            else:
                registration_available_date = ""
            rows.append(
                {
                    "election_id": election_id,
                    "sg_id": sg_id,
                    "sg_typecode": "1",
                    "candidate_id": _clean(record.get("huboid")),
                    "candidate_number": _clean(record.get("giho")),
                    "candidate_name": _clean(record.get("name")),
                    "birthday": _clean(record.get("birthday")),
                    "gender": _clean(record.get("gender")),
                    "party_name": _clean(record.get("jdName")),
                    "job": _clean(record.get("job")),
                    "education": _clean(record.get("edu")),
                    "career1": _clean(record.get("career1")),
                    "career2": _clean(record.get("career2")),
                    "registration_date": registration_available_date,
                    "registration_available_date": registration_available_date,
                    "status_as_of_fetch": _clean(record.get("status")),
                    "status_observed_at": collected_at,
                    "status_point_in_time_eligible": False,
                    "source_url": (
                        f"{CANDIDATE_REGISTRY_BASE}/"
                        "getPofelcddRegistSttusInfoInqire"
                    ),
                    "source_record_sha256": _record_hash(record),
                    "derivation_version": "nec_presidential_candidate_registry_v1",
                }
            )
    roster = pd.DataFrame(rows).sort_values(
        ["election_id", "candidate_number", "candidate_name"]
    ).reset_index(drop=True)
    _atomic_csv(args.output_dir / "nec_presidential_candidate_registry.csv", roster)
    _atomic_json(
        args.output_dir / "nec_presidential_candidate_registry_manifest.json",
        {
            "schema": "nec_presidential_candidate_registry_manifest_v1",
            "collected_at": collected_at,
            "elections": list(ELECTIONS),
            "maximum_election": "pres_2022",
            "post_2022_query_count": 0,
            "outcome_fields_used": [],
            "rows": len(roster),
            "pages": len(page_provenance),
            "status_policy": (
                "Current registration status is retained as an audit fact only; "
                "it is not PIT-eligible until an official event date is attached."
            ),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(roster),
                "elections": roster["election_id"].nunique(),
                "post_2022_query_count": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
