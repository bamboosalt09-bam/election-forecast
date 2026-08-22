# Third-candidate ceiling: stage-independent recipient weights

## Status

- Date: 2026-08-22
- Status: implemented and measured; **not adopted**. The default stays `live`.
- Post-2022 outcomes used: none
- Reproduce with `python scripts/evaluate_lineage_recipient_mode.py`

## The concern

`apply_lineage_ceiling` takes mass off a self-founded third candidate and hands
it to the two majors, split in proportion to `out.loc[majors.index, output_column]`
- the prediction as it stands when the ceiling runs. Whatever an earlier
postprocess did to the two majors therefore decides the split.

On the 2025 target that compounds. The strong incumbent veto widens the
two-major gap from 7.80 to 17.48 points, and the ceiling then redistributes
13.64 points of recovered mass at that widened ratio, adding a further 3.18.
Each rule is defensible alone; the composition amplifies.

## The change

A `recipient_weight_mode` parameter, matching the shape
`weak_same_lane_refusal` already uses:

- `live` (default) - split at the current prediction, as before
- `reference` - split at `anchored_pred`, a column no postprocess writes to

The cap itself is unchanged in both modes; only the split of the excess moves.
A missing or degenerate reference falls back to the live split rather than
skipping the cap, so the third candidate is never left above its own ceiling.

## Result

| mode | regional weighted macro | national macro | winners |
| --- | ---: | ---: | ---: |
| **live** (shipped) | **2.712233** | **0.720994** | 4/5 |
| reference | 2.718386 | 0.727149 | 4/5 |

By election, national candidate MAE:

| election | live | reference |
| --- | ---: | ---: |
| pres_2002 | **2.3416** | 2.3723 |
| pres_2007 | 0.6610 | 0.6610 |
| pres_2012 | 0.1271 | 0.1271 |
| pres_2017 | 0.2011 | 0.2011 |
| pres_2022 | 0.2741 | 0.2741 |

On the 2025 target, as an arithmetic reconstruction from the stage audit with
no outcome read:

| mode | slot A | slot B | two-major gap |
| --- | ---: | ---: | ---: |
| live | 34.60 | 55.20 | 20.60 |
| reference | 36.67 | 53.13 | **16.46** |

## Why it is not adopted

The change does exactly what it was designed to do: on 2025 it removes the
inherited-ratio amplification and narrows the gap by 4.15 points. The scored
panel disagrees, very slightly, and **the disagreement is one observation**. The
ceiling binds for 권영길 2002 alone, so the entire difference is 0.0307 %p on a
single election.

One observation cannot adjudicate a structural choice in either direction.
Adopting on the mechanism argument would be changing the model on an
unobservable; rejecting on 0.0307 %p would be reading noise as evidence. The
honest position is that the panel is silent, so the shipped behaviour stands and
the alternative is kept available and documented.

This is the same limit `docs/EXPERIMENT_POSTPROCESS_ABLATION_20260822.md`
records from the other direction: the configuration that produces the 2025
amplification - all three layers firing together - has zero scored
observations.

## What would settle it

Nothing available in the current panel. A scored election in which the veto and
the ceiling both fire would decide it; there is none, and there is no way to
manufacture one. This belongs on the pre-registration list: fix the mode now,
declare it, and let the next election test it.
