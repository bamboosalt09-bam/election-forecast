<!-- active-model-version: v31 -->
# Two attempts at the regionalism gap, both abandoned

## Status

- Date: 2026-08-25
- Status: **both rejected**; no model, prediction or frozen artifact changed
- Written so neither is re-tried without first reading what was measured

V31 stopped the transform from publishing exactly zero. It did not answer the
larger question behind that defect: whether regional variation is
under-reflected, particularly for candidates the model has little history on.
Two routes were tried and both were abandoned, for different reasons.

---

## 1. Inheriting a candidate's own party lineage

### The idea

`party_lineage()` maps a **party name** to a lineage and takes no candidate. A
candidate who leaves their party and runs unaffiliated therefore resolves to
`independent` and loses the regional profile their lineage has. 이회창 2007 is
the case: 한나라당 in 2002, unaffiliated in 2007, and every regional signal
collapsed.

| field | 이회창 2002 | 이회창 2007 |
| --- | ---: | ---: |
| lineage | mainstream_conservative | independent |
| candidate ballot history | 6.002 | **0.000** |
| party history | 1.000 | **0.000** |
| recent bloc base | 0.482 | **0.000** |
| camp regional mean | 0.391 | **0.000** |

`candidate_camp` stayed `camp_conservative` throughout, and his realised
regional spread is 1.944× the predicted one — the worst in the panel.

The fix looked like a lookup correction rather than a new component: when the
ballot party has no lineage, take the one the same candidate carried in their
most recent earlier candidacy. Only strictly earlier elections are read, party
affiliation is public before an election, and no constant is introduced.

### Why it was abandoned

It was built, and it changed nothing. Predictions came out identical to V31 to
the last decimal.

The lineage label did change — `independent` → `mainstream_conservative` — but
nothing downstream consumes it. The collapsed regional signals are keyed on
**`bloc`**, not on the lineage id, and `bloc` was still 무소속.

Worse, the active configuration sets

```json
"direct_score_scope": "non_major_only"
```

so the direct lineage score deliberately **excludes** the three mainstream
lineages. Inheriting `mainstream_conservative` would have moved 이회창 2007
from "no lineage found" into "lineage found and therefore excluded". The two
paths reach the same zero for opposite reasons.

The exclusion is not an oversight; the code states its reason — the broad camp
prior has already priced regional deficits, and applying negative lineage gaps
here would count weak-terrain evidence twice.

### What would be needed instead

The real collapse is in the `bloc` join, and moving 이회창 to `bloc = 보수`
gives him and 이명박 the same conservative base row. **No election in the panel
has two candidates sharing a bloc**, so the structure assumes one, and 2007's
conservative split is the case it does not represent. Splitting that base
requires a rule for how, which is the arbitrariness this project has been
declining elsewhere.

### Correction to an earlier claim

An earlier note in this session said `lineage_identity_score` is zero in 230 of
232 rows. That was a pres_2022-only measurement generalised to the panel. Across
the panel it is nonzero in 35 of 232 rows, and
`lineage_identity_log_shift` — what actually moves predictions — is nonzero in
**87 of 232**, with a maximum effect of `+8.68%` on a share. The layer is
narrowly scoped, not dead.

---

## 2. Damping the expansion at extreme deviations

### The idea

V31's multiplicative form guarantees a positive share but scales deviations
**uniformly** in log space, so an already-extreme low is pushed lower. The 2025
demonstration shows it: 김문수's 광주 entered the transform at `2.670%` and left
at `2.053%`, against a realised `8.104%`. Across all 51 forecast rows the
expansion moved 21 toward the result and 30 away.

The proposal was an exponent that falls as the deviation grows, pulling extreme
values back toward the centre with a cap on how much regression is allowed, and
fitted **inside each fold from all strictly earlier elections** (rolling) so the
scored panel selects nothing.

### Why it was abandoned

The rolling evidence points the other way.

Writing `z = log(share / candidate's national level)`, and splitting the panel
by the sign of the predicted deviation:

| band | n | mean `z_pred` | mean `z_act` | slope |
| --- | ---: | ---: | ---: | ---: |
| **very low (z < −1.0)** | 17 | **−1.700** | **−1.828** | 0.745 |
| low (−1.0 … −0.35) | 19 | −0.627 | −0.524 | 1.379 |
| centre | 54 | −0.002 | −0.006 | 0.676 |
| mild high (0.05 … 0.35) | 59 | +0.176 | +0.138 | 0.532 |
| high (0.35 … 1.0) | 30 | +0.542 | +0.350 | 1.989 |

**In the scored panel, when the model places a candidate far below their own
national level, reality is usually further below still** — 12 of the 17 rows,
with 홍준표's 호남 in 2017 at ratios of 1.26, 1.36 and 1.44.

A damping factor fitted on rolling prior elections would therefore come out
**above 1** in the low tail: amplification, not regression toward the centre.
That is the opposite of the intent.

2025 is the only place the pattern reverses:

| very-low band | mean `z_pred` | mean `z_act` | ratio |
| --- | ---: | ---: | ---: |
| panel 2002–2022 | −1.700 | −1.828 | **1.096** (more extreme) |
| 2025 | −2.221 | −1.513 | **0.688** (less extreme) |

So the only evidence for damping is the one election that cannot be fitted
without becoming outcome-informed. Building it would mean choosing a shape
because 2025 wants it, which is the definition of the overfitting this project
declines. Abandoned.

---

## What is left open

The 2025 error is dominated by a **national level six points low** for the
conservative candidate — his largest regional misses include 대구, his
strongest region, not only 호남. Terminal dispersion machinery operates around
a candidate's national level and cannot repair a level that is wrong.

Whether a shallow-history candidate's level is systematically mis-set is the
open question, and it is not answerable from one election.

## Related

- `EXPERIMENT_V31_MULTIPLICATIVE_EXPANSION_20260825.md` — the change that stands
- `PRES_2025_V31_POST_ELECTION_EVALUATION.md` — the 2025 measurements quoted here
- `METRIC_WEIGHTING_20260825.md` — the other proposal investigated and rejected
