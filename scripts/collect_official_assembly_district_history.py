"""Collect NEC Assembly constituency results through 2020 with raw caching."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from news_collector.sources.nec_vote_api import (  # noqa: E402
    COUNT_OPERATION,
    NEC_VOTE_BASE,
)
from news_collector.sources.public_data_api import PublicDataApiClient  # noqa: E402
from presidential_issue_engine.assembly_district_history import (  # noqa: E402
    build_assembly_district_history,
)


ELECTIONS = {
    "assembly_1992_district": "19920324",
    "assembly_1996_district": "19960411",
    "assembly_2000_district": "20000413",
    "assembly_2004_district": "20040415",
    "assembly_2008_district": "20080409",
    "assembly_2012_district": "20120411",
    "assembly_2016_district": "20160413",
    "assembly_2020_district": "20200415",
}
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "official_sources"


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
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

    client = PublicDataApiClient(
        base_url=NEC_VOTE_BASE,
        cache_dir=args.output_dir / "cache" / "nec_assembly_district_counts",
        timeout=45.0,
    )
    frames: list[pd.DataFrame] = []
    page_count = 0
    for election_id, sg_id in ELECTIONS.items():
        records, provenance = client.fetch_all(
            COUNT_OPERATION,
            params={"sgId": sg_id, "sgTypecode": "2"},
            num_rows=100,
            offline=args.offline,
            refresh=args.refresh,
        )
        page_count += len(provenance)
        frames.append(
            build_assembly_district_history(
                records,
                election_id=election_id,
                election_date=pd.Timestamp(sg_id),
            )
        )
    history = pd.concat(frames, ignore_index=True)
    output = args.output_dir / "nec_assembly_district_history.csv"
    _atomic_csv(output, history)
    manifest = {
        "schema": "nec_assembly_district_collection_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elections": list(ELECTIONS),
        "maximum_election": "assembly_2020_district",
        "post_2022_queries": 0,
        "outcome_fields_used_for_presidential_target": [],
        "rows": len(history),
        "districts": int(
            history[["election_id", "region_id", "district_name"]]
            .drop_duplicates()
            .shape[0]
        ),
        "pages": page_count,
    }
    _atomic_json(
        args.output_dir / "nec_assembly_district_history_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
