# common — shared contracts

`common` is the thin layer the two competition projects agree on. Keeping it
thin is the point: the moment engine logic leaks in here, the two projects start
constraining each other.

## What is shared (here)

| package | holds | why shared |
| --- | --- | --- |
| `shared_schema` | `ElectionType`, `AggregationRule`, `ContestRow` | so the open-source engine can grow presidential → 총선 → 지선 on the *same* schema the stats project uses (stats just pins everything to presidential defaults) |
| `election_slot_schema` | the A/B/C/alpha slot definition + validation | both projects compare candidates as slots, not ideologies |
| `feature_schema` | `FeatureRow` — the numericalization contract | every scorer (rule-based or AI) emits this shape; the engine binds to the schema, never to a specific scorer |
| `evaluation` | MAE, %p error, winner accuracy | one ruler, so "complex engine + AI" can be compared to "simple engine + rules" honestly |

## What is deliberately NOT shared

- **Numericalization** — *how* raw news/editorials/polls become numbers.
  - stats: rule-based word/expression frequency + negative-keyword ratio (AI-free).
  - oss: may additionally use an open-weight model for stance/frame/issue-linkage.
- **Forecast engine** — stats uses a simple weighted-sum + softmax + Monte Carlo
  baseline; oss may use a more complex model and per-election-type aggregation.

## The mental model

```
[raw]  →  ⟦ numericalization (rule | AI) ⟧  →  [ feature_schema ]  →  ⟦ engine (simple | complex) ⟧  →  ⟦ evaluation ⟧  →  [%p error]
              DIFFERS per competition            SHARED (common)         DIFFERS per competition          SHARED (common)
```

## Relationship to existing code

`common` is a clean extraction of contracts that already exist inside
`src/election_forecast/presidential` (the 7 political variables, the A/B/C/alpha
slots, the evaluation metrics). It adds the `election_type` / `contest_id` /
`aggregation_rule` dimensions on top. The open-source engine is expected to
migrate to importing `common` so there is one source of truth; until that
migration lands, `common` and the presidential module intentionally mirror each
other.
