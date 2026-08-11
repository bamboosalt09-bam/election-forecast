"""Long-running assembly speech issue reprocessing.

This script is checkpointed because the source archive is several GB and the
full pass can take a long time. It regenerates:

- data/issue_salience_assembly.csv
- data/candidate_issue_link.csv

The actual keyword/phrase matching is delegated to ``assembly_batch``, which
uses the open-source phrase-aware issue matcher.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_collector.sources.assembly_batch import ELECTION_WINDOWS, accumulate, iter_rows_from_xlsx  # noqa: E402
from news_collector.sources.contextual_issue_weights import (  # noqa: E402
    housing_issue_boosts_from_index,
    load_issue_context_rules,
)
from news_collector.sources.datalab import load_issue_keywords  # noqa: E402
from news_collector.sources.issue_term_weights import load_campaign_issue_terms, merge_issue_terms  # noqa: E402
from news_collector.sources.member_party import party_bloc  # noqa: E402
from news_collector.sources.salience_base import CANONICAL_COLUMNS, normalize_within_election  # noqa: E402

DEFAULT_ZIP = Path(r"C:\Users\최성준\Downloads\trash_dataset.zip")
KEYWORDS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "issue_keywords.csv"
CAMPAIGN_TERMS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "campaign_issue_terms.csv"
MEGA_TERMS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "mega_issue_terms.csv"
CONTEXT_RULES = ROOT / "presidential_issue_engine" / "fixed_dataset" / "issue_context_rules.csv"
HOUSING_INDEX = ROOT / "presidential_issue_engine" / "fixed_dataset" / "housing_price_index_sido.csv"
ROSTER = ROOT / "data" / "assembly_roster.csv"
RESULTS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv"
OUT_SALIENCE = ROOT / "data" / "issue_salience_assembly.csv"
OUT_LINK = ROOT / "data" / "candidate_issue_link.csv"
STATE_DIR = ROOT / "data" / "processed"
CHECKPOINT = STATE_DIR / "assembly_reprocess_checkpoint.json"
LOG_PATH = ROOT / "logs" / "assembly_reprocess.log"

ELECTION_TO_ASSEMBLY = {
    "pres_2002": "16",
    "pres_2007": "17",
    "pres_2012": "19",
    "pres_2017": "20",
    "pres_2022": "21",
}
ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


@dataclass(frozen=True)
class WorkItem:
    """One outer or nested xlsx workbook inside the source archive."""

    election_id: str
    assembly: str
    key: str
    outer_name: str
    inner_name: str | None


def log(message: str) -> None:
    """Append a timestamped progress line to stdout and the run log."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{pd.Timestamp.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def counter_to_json(counter: Counter) -> list[list[object]]:
    """Serialize a Counter whose keys are tuples."""

    return [[*key, float(value)] for key, value in counter.items()]


def counter_from_json(rows: Iterable[list[object]]) -> Counter:
    """Deserialize a Counter whose keys are tuples."""

    return Counter({tuple(row[:-1]): float(row[-1]) for row in rows})


def load_state(reset: bool) -> tuple[set[str], Counter, Counter]:
    """Load checkpoint state unless a clean restart was requested."""

    if reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    if not CHECKPOINT.exists():
        return set(), Counter(), Counter()
    with CHECKPOINT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return (
        set(payload.get("completed_keys", [])),
        counter_from_json(payload.get("sal_counts", [])),
        counter_from_json(payload.get("mem_counts", [])),
    )


def save_state(completed_keys: set[str], sal_counts: Counter, mem_counts: Counter) -> None:
    """Persist checkpoint state atomically after each workbook."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    payload = {
        "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "completed_keys": sorted(completed_keys),
        "sal_counts": counter_to_json(sal_counts),
        "mem_counts": counter_to_json(mem_counts),
    }
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    tmp.replace(CHECKPOINT)


def iter_work_items(zip_path: Path) -> list[WorkItem]:
    """List all target xlsx workbooks for presidential-window assemblies."""

    items: list[WorkItem] = []
    with zipfile.ZipFile(zip_path) as master:
        for outer in master.infolist():
            if outer.is_dir():
                continue
            for election_id, assembly in ELECTION_TO_ASSEMBLY.items():
                if f"제{assembly}대" not in outer.filename:
                    continue
                lower = outer.filename.lower()
                if lower.endswith(".xlsx"):
                    items.append(WorkItem(election_id, assembly, outer.filename, outer.filename, None))
                elif lower.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(master.read(outer))) as inner:
                        for nested in inner.infolist():
                            if nested.is_dir() or not nested.filename.lower().endswith(".xlsx"):
                                continue
                            key = f"{outer.filename}::{nested.filename}"
                            items.append(WorkItem(election_id, assembly, key, outer.filename, nested.filename))
    return items


def read_workbook_bytes(master: zipfile.ZipFile, item: WorkItem) -> bytes:
    """Read one outer or nested workbook from the source archive."""

    if item.inner_name is None:
        return master.read(item.outer_name)
    with zipfile.ZipFile(io.BytesIO(master.read(item.outer_name))) as inner:
        return inner.read(item.inner_name)


def member_to_bloc_for(roster: pd.DataFrame, assembly: str) -> dict[str, str]:
    """Map member names to broad party blocs for one National Assembly term."""

    rows = roster[roster["daesu"].astype(str) == str(assembly)]
    return {
        str(name).strip(): str(bloc).strip()
        for name, bloc in zip(rows["name"], rows["bloc"])
        if str(name).strip() and str(bloc).strip()
    }


def slot_map_for_election(results: pd.DataFrame, election_id: str) -> dict[str, str]:
    """Map broad party blocs to presidential candidate slots."""

    rows = results[(results.election_id == election_id) & (results.slot.isin(["A", "B", "C"]))]
    bloc_to_slot: dict[str, str] = {}
    for row in rows.itertuples(index=False):
        if row.slot == "C" and not bool(row.is_active_slot):
            continue
        bloc_to_slot[party_bloc(row.party_name)] = row.slot
    return bloc_to_slot


def write_outputs(sal_counts: Counter, mem_counts: Counter, results: pd.DataFrame) -> None:
    """Write canonical salience and candidate-link CSV outputs."""

    salience_frames = []
    for election_id in ELECTION_TO_ASSEMBLY:
        rows = [
            {"issue_name": issue, "period": period, "count": count}
            for (eid, issue, period), count in sal_counts.items()
            if eid == election_id
        ]
        if rows:
            salience_frames.append(normalize_within_election(pd.DataFrame(rows), "count", election_id, "assembly_speech"))
    if salience_frames:
        salience = pd.concat(salience_frames, ignore_index=True)
    else:
        salience = pd.DataFrame(columns=CANONICAL_COLUMNS)

    slot_maps = {election_id: slot_map_for_election(results, election_id) for election_id in ELECTION_TO_ASSEMBLY}
    link_rows = []
    for (election_id, bloc, issue, _period), mentions in mem_counts.items():
        slot = slot_maps.get(election_id, {}).get(bloc, "alpha")
        link_rows.append({"election_id": election_id, "slot": slot, "issue_name": issue, "mentions": mentions})
    if link_rows:
        link = pd.DataFrame(link_rows).groupby(["election_id", "slot", "issue_name"], as_index=False)["mentions"].sum()
        link["emphasis_volume"] = link.groupby("election_id")["mentions"].transform(lambda s: (s / s.max()).round(4))
        link["emphasis_within"] = link.groupby(["election_id", "slot"])["mentions"].transform(lambda s: (s / s.sum()).round(4))
    else:
        link = pd.DataFrame(columns=["election_id", "slot", "issue_name", "mentions", "emphasis_volume", "emphasis_within"])

    salience.to_csv(OUT_SALIENCE, index=False, encoding="utf-8-sig")
    link.to_csv(OUT_LINK, index=False, encoding="utf-8-sig")
    sal_elections = salience["election_id"].nunique() if not salience.empty else 0
    link_elections = link["election_id"].nunique() if not link.empty else 0
    log(f"wrote {OUT_SALIENCE.relative_to(ROOT)} rows={len(salience)} elections={sal_elections}")
    log(f"wrote {OUT_LINK.relative_to(ROOT)} rows={len(link)} elections={link_elections}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="source trash_dataset.zip path")
    parser.add_argument("--reset", action="store_true", help="discard existing checkpoint and start from scratch")
    return parser.parse_args()


def main() -> None:
    """Run the long assembly issue reprocessing job."""

    args = parse_args()
    zip_path = args.zip.resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"source zip not found: {zip_path}")

    log(f"START zip={zip_path}")
    keyword_map = load_issue_keywords(KEYWORDS)
    campaign_terms, campaign_weights = load_campaign_issue_terms(CAMPAIGN_TERMS, ELECTION_TO_ASSEMBLY)
    mega_terms, mega_weights = load_campaign_issue_terms(MEGA_TERMS, ELECTION_TO_ASSEMBLY)
    issue_boosts, boost_diagnostics = housing_issue_boosts_from_index(
        HOUSING_INDEX,
        ELECTION_DATES,
        ELECTION_TO_ASSEMBLY,
    )
    context_rules = load_issue_context_rules(CONTEXT_RULES, ELECTION_TO_ASSEMBLY)
    keyword_maps = {
        election_id: merge_issue_terms(
            merge_issue_terms(keyword_map, campaign_terms.get(election_id, {})),
            mega_terms.get(election_id, {}),
        )
        for election_id in ELECTION_TO_ASSEMBLY
    }
    term_weights = {
        election_id: {
            **campaign_weights.get(election_id, {}),
            **mega_weights.get(election_id, {}),
        }
        for election_id in ELECTION_TO_ASSEMBLY
    }
    roster = pd.read_csv(ROSTER, dtype=str).fillna("")
    results = pd.read_csv(RESULTS)
    completed_keys, sal_counts, mem_counts = load_state(reset=args.reset)
    items = iter_work_items(zip_path)
    member_maps = {assembly: member_to_bloc_for(roster, assembly) for assembly in ELECTION_TO_ASSEMBLY.values()}
    campaign_count = sum(len(weights) for weights in campaign_weights.values())
    mega_count = sum(len(weights) for weights in mega_weights.values())
    context_rule_count = sum(len(rules) for rules in context_rules.values())
    log(
        f"workbooks={len(items)} completed={len(completed_keys)} "
        f"issues={len(keyword_map)} campaign_weights={campaign_count} "
        f"mega_weights={mega_count} context_rules={context_rule_count} reset={args.reset}"
    )
    if not boost_diagnostics.empty:
        for row in boost_diagnostics.itertuples(index=False):
            log(
                f"housing_boost {row.election_id} mean_change={row.mean_housing_change_pct:.2f}% "
                f"multiplier={row.housing_issue_multiplier:.3f}"
            )

    started = time.monotonic()
    with zipfile.ZipFile(zip_path) as master:
        for idx, item in enumerate(items, 1):
            if item.key in completed_keys:
                continue
            before_sal = sum(sal_counts.values())
            before_mem = sum(mem_counts.values())
            try:
                data = read_workbook_bytes(master, item)
                accumulate(
                    iter_rows_from_xlsx(data),
                    keyword_maps[item.election_id],
                    {item.election_id: ELECTION_WINDOWS[item.election_id]},
                    member_maps[item.assembly],
                    sal_counts,
                    mem_counts,
                    term_weights_by_election={item.election_id: term_weights.get(item.election_id, {})},
                    issue_boosts_by_election={item.election_id: issue_boosts.get(item.election_id, {})},
                    context_rules_by_election={item.election_id: context_rules.get(item.election_id, [])},
                )
                completed_keys.add(item.key)
                save_state(completed_keys, sal_counts, mem_counts)
                log(
                    f"[{idx}/{len(items)}] done {item.election_id} {Path(item.key).name[:90]} "
                    f"sal+{sum(sal_counts.values()) - before_sal} mem+{sum(mem_counts.values()) - before_mem} "
                    f"elapsed_min={(time.monotonic() - started) / 60:.1f}"
                )
            except Exception as exc:  # noqa: BLE001 - keep the long batch moving
                completed_keys.add(item.key)
                save_state(completed_keys, sal_counts, mem_counts)
                log(f"[{idx}/{len(items)}] skip {item.election_id} {Path(item.key).name[:90]} error={exc!r}")

    write_outputs(sal_counts, mem_counts, results)
    log("DONE")


if __name__ == "__main__":
    main()
