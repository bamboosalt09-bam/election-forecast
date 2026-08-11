"""Bridge crawled news metadata → issue_store salience rows.

The collector (`news_collector`) fills `raw_lake` with article *metadata*
(title/summary/date, no body). This module turns that metadata into the issue
memory: it counts how loudly each issue appeared in the news per time bucket and
emits rows conforming to ``common.issue_store.IssueEventRow`` with
``populator="aggregate"``.

Source-agnostic: identical treatment for GDELT, Naver API, or BIGKinds imports.
No AI — pure keyword counting per ``data/config/base_issue_keywords.csv``:

- salience_score   = bucket article count / max bucket count in period (min-max).
- candidate_link   = (issue ∧ slot-candidate co-mentions) / slot-candidate mentions.
- direction_score  = lexicon ratio (pos - neg) / (pos + neg) over matched text.

Output columns mirror ``common.issue_store.IssueEventRow`` exactly so the rollup
and engine consume it unchanged.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import pandas as pd

# Conforms to common.issue_store.IssueEventRow (kept inline so this src/ package
# stays installable without depending on the repo-root `common` package; the
# test suite validates conformance against common.issue_store).
ISSUE_EVENT_COLUMNS = [
    "issue_id", "election_id", "issue_name", "issue_type", "event_date", "available_date",
    "slot", "candidate_id", "region_scope", "salience_score", "direction_score",
    "candidate_link_score", "media_reliability_score", "final_issue_score",
    "populator", "confidence", "source_note",
]


def load_keyword_map(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Read issue_keywords.csv → {issue_name: {keywords, pos, neg}}."""

    out: dict[str, dict[str, list[str]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("issue_name") or "").strip()
            if not name:
                continue
            out[name] = {
                "keywords": _split(row.get("keywords")),
                "pos": _split(row.get("pos_keywords")),
                "neg": _split(row.get("neg_keywords")),
            }
    return out


def load_slot_names(path: str | Path) -> dict[str, list[str]]:
    """Read candidate_slots.csv → {slot: [candidate_name, ...]} for co-mention."""

    out: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            slot = (row.get("slot") or "").strip()
            name = (row.get("candidate_name") or "").strip()
            if slot and name:
                out.setdefault(slot, []).append(name)
    return out


def read_raw_lake(input_dir: str | Path) -> pd.DataFrame:
    """Read append-only RawArticle JSONL from raw_lake into a frame."""

    rows: list[dict] = []
    for path in glob.glob(str(Path(input_dir) / "**" / "*.jsonl"), recursive=True):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "title": obj.get("title") or "",
                        "summary": obj.get("summary") or "",
                        "available_date": obj.get("available_date") or obj.get("published_at"),
                    }
                )
    return pd.DataFrame(rows)


def _split(value: str | None) -> list[str]:
    return [t.strip() for t in (value or "").split("|") if t.strip()]


def _matches(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t and t in text)


def aggregate_salience(
    articles: pd.DataFrame,
    keyword_map: dict[str, dict[str, list[str]]],
    slot_names: dict[str, list[str]],
    election_id: str,
    bucket: str = "W",
    region_scope: str = "ALL",
    taxonomy_type: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Count issue exposure per (issue, slot, time-bucket) → issue_events rows.

    ``taxonomy_type`` optionally maps issue_name → issue_type (from
    issue_taxonomy.csv); defaults to ``policy``.
    """

    if articles.empty:
        return pd.DataFrame(columns=ISSUE_EVENT_COLUMNS)
    df = articles.copy()
    df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).astype(str)
    df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce")
    df = df.dropna(subset=["available_date"])
    if df.empty:
        return pd.DataFrame(columns=ISSUE_EVENT_COLUMNS)
    df["bucket"] = df["available_date"].dt.to_period(bucket).dt.end_time.dt.date

    records: dict[tuple, dict] = {}
    slot_mentions: dict[str, int] = {s: 0 for s in slot_names}
    for _, art in df.iterrows():
        text = art["text"]
        present_slots = [s for s, names in slot_names.items() if _matches(text, names) > 0]
        for s in present_slots:
            slot_mentions[s] += 1
        for issue, kw in keyword_map.items():
            if _matches(text, kw["keywords"]) == 0:
                continue
            pos, neg = _matches(text, kw["pos"]), _matches(text, kw["neg"])
            direction = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
            # Attribute to co-mentioned slots; if none, to 'alpha' (general).
            targets = present_slots or ["alpha"]
            for s in targets:
                key = (issue, s, art["bucket"])
                rec = records.setdefault(
                    key, {"count": 0, "dir_sum": 0.0, "link": 0, "type": (taxonomy_type or {}).get(issue, "policy")}
                )
                rec["count"] += 1
                rec["dir_sum"] += direction
                if s in present_slots:
                    rec["link"] += 1

    if not records:
        return pd.DataFrame(columns=ISSUE_EVENT_COLUMNS)
    max_count = max(r["count"] for r in records.values()) or 1
    out_rows: list[dict] = []
    for (issue, slot, wk), rec in records.items():
        out_rows.append(
            {
                "issue_id": f"agg_{election_id}_{issue}_{slot}_{wk}",
                "election_id": election_id,
                "issue_name": issue,
                "issue_type": rec["type"],
                "event_date": str(wk),
                "available_date": str(wk),
                "slot": slot,
                "candidate_id": None,
                "region_scope": region_scope,
                "salience_score": round(rec["count"] / max_count, 4),
                "direction_score": round(rec["dir_sum"] / rec["count"], 4),
                "candidate_link_score": round(rec["link"] / slot_mentions[slot], 4)
                if slot_mentions.get(slot) else 0.0,
                "media_reliability_score": 1.0,
                "final_issue_score": None,
                "populator": "aggregate",
                "confidence": 1.0,
                "source_note": "salience aggregated from crawled metadata",
            }
        )
    return pd.DataFrame(out_rows, columns=ISSUE_EVENT_COLUMNS)


def build_issue_events(
    input_dir: str | Path,
    keywords_path: str | Path,
    slots_path: str | Path,
    election_id: str,
    out_path: str | Path,
    taxonomy_path: str | Path | None = None,
) -> Path:
    """End-to-end: raw_lake → issue_events.csv (populator=aggregate)."""

    articles = read_raw_lake(input_dir)
    kw = load_keyword_map(keywords_path)
    slots = load_slot_names(slots_path)
    tax = None
    if taxonomy_path and Path(taxonomy_path).exists():
        tdf = pd.read_csv(taxonomy_path)
        tax = dict(zip(tdf["issue_name"], tdf.get("default_issue_type", "policy")))
    frame = aggregate_salience(articles, kw, slots, election_id, taxonomy_type=tax)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    return out
