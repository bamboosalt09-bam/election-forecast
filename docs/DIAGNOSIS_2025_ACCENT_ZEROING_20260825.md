<!-- active-model-version: v31 -->
# The 2025 forecast runs without its regional accent layer

## Status

- Date: 2026-08-25
- Status: **diagnosed, not fixed**
- No model, prediction or frozen artifact changed by this document
- The defect is in the prospective path only; the scored panel is unaffected

## What is wrong

The published 2025 D-1 forecast is produced with the regional accent layer
contributing **nothing**. 31 of the 32 accent columns are identically zero
across all 51 rows, `regional_accent_log_shift` is exactly `0` where the scored
elections carry up to `0.095`, and the layer's gain for `pres_2025` is `0.0`
where the scored elections get `0.10`–`0.20`.

This is not a property of forecasting. The evidence the layer needs exists and
is computed correctly for 2025.

## The exact line

`scripts/run_prospective_forecast.py`, in `_target_base`:

```python
# The assembled target already contains the same electorate and issue
# feature contract.  Missing diagnostic-only columns are inert.
for column in historical_base.columns:
    if column not in out.columns:
        out[column] = np.nan if column == "actual" else 0.0
```

The assembled 2025 target frame carries 259 columns; the historical base frame
carries 111. **Forty** of the base's columns are absent from the target and get
`0.0`, and **27 of those 40 are the entire `regional_accent_*` family**.

The comment states the assumption that makes this safe — those columns are
"diagnostic-only" and therefore "inert". That is true of most of the forty. It
is false for the accent family, which is load-bearing:

```
regional_accent_{axis}_reliability  ->  candidate regional_accent_reliability
                                    ->  regional_accent_gain_by_target
                                    ->  regional_accent_log_shift  ->  prediction
```

Zeroing the first element of that chain sets the last to zero.

## How it was localised

Each step was measured rather than inferred, and three earlier explanations
were wrong and are recorded here so they are not tried again.

| stage | pres_2022 | pres_2025 |
| --- | ---: | ---: |
| `_regional_accent_summary` returns | 17 rows / 357 non-zero | **17 rows / 357 non-zero** |
| `estimate_electorate_layers` returns `regional_accent_reliability` | 51/51 | **51/51** |
| all 27 accent columns at the estimator | populated | **populated, identically** |
| arriving at `_attach_layers` | 51/51 | **0/51** |
| frame reaching the gain rule | 51 positive rows | **0 positive rows** |
| resulting gain | `0.20` | **`0.0`** |
| `regional_accent_log_shift` | max `0.0949` | **`0`** |

### Three explanations that were wrong

1. **"The layer does not run for 2025."** It runs. The estimator produces
   pres_2025 identically to pres_2022, column for column.
2. **"The gain map iterates only the scored elections, so 2025 has no entry."**
   In the prospective run the map *does* contain `pres_2025`, with value `0.0`.
3. **"Fixing the gain-map iteration fixes it."** It does not. The rule reads
   `regional_accent_reliability` from a frame where that column is already
   zero, so the same rule returns `0.0` either way. An earlier probe appeared
   to confirm this fix by feeding the shipped audit CSV — which already
   contains the zeros — into the rule, and proved nothing.

## Error, not derivation

Four things separate this from a deliberate exclusion:

- the accent evidence is computable for 2025 from **pre-cutoff** history alone,
  and is computed;
- `FORBIDDEN_OUTCOME_COLUMNS`, the guard that blocks outcome-derived inputs,
  does not name any accent column, and it raises rather than blanking;
- no code or document declares the accent layer inapplicable to a prospective
  target;
- the zeroing is generic — it catches whatever happens to be missing — and is
  justified by a comment whose stated assumption does not hold for what it
  caught.

## The zero-fill is the symptom, not the source

An attempt to recover the columns at `_target_base` — copying them from the
assembled target instead of zeroing them — found **0 of the 27 present**. The
2025 target frame never carries the accent family at all, so there is nothing
for the loop to preserve. It zeroes them because they were never built for
these rows.

The `estimate_electorate_layers` call that produces a populated pres_2025
accent is a different call on a different frame, and its output is not what
reaches `_target_base`. So the fix is not a column pass-through: the 2025
target rows have to be routed through the same layer estimation the scored rows
get.

That is a larger change than the zero-fill line suggests, and it is why this
document stops at diagnosis.

## What a fix has to establish

Restoring the accent for 2025 changes a published forecast, so it belongs to a
new version rather than an in-place edit. Before that, two things need
measuring:

1. whether the scored panel is byte-identical under the fix — it should be,
   since the five scored gains are computed from their own rows, but this must
   be shown rather than assumed;
2. how far the 2025 forecast moves, per region and per candidate.

The zero-fill itself should also stop being silent. A loop that fills whatever
is missing cannot know what is inert; naming the columns it is allowed to fill,
and failing on anything else, converts this class of defect from invisible to
loud.

## A fourth wrong turn, recorded

The first attempt at the fix rebuilt the accent by calling
``_attach_candidate_regional_accent`` directly on the target frame. It produced
values that *looked* right - 51/51 populated, mean reliability 0.6365 against
the panel's 0.6727 - and were wrong, differing from the canonical computation
by up to `0.0357` in reliability, because inside the estimator that attachment
runs after `_candidate_camp_frame` and the rest of the assembly.

This document had already said not to do that:

> the fix is not a column pass-through: the 2025 target rows have to be routed
> through the same layer estimation the scored rows get.

The correct builder calls `estimate_electorate_layers` and takes its output, so
it is equal to the canonical path by construction rather than by resemblance.
Verified against a traced canonical run: maximum difference `5.6e-17`.

A plausible-looking number is the failure mode here. Mean reliability of 0.6365
next to a panel mean of 0.6727 reads as confirmation; only comparing against
the canonical values showed it was not.

## Related

- `DIAGNOSIS_REGIONALISM_DEAD_ENDS_20260825.md` — the regionalism work this came
  out of
- `PRES_2025_V31_POST_ELECTION_EVALUATION.md` — the 2025 figures quoted here
