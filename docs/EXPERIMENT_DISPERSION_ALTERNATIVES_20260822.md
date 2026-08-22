# Four ways to correct regional dispersion, and what they share

## Status

- Date: 2026-08-22
- Status: four mechanisms built and measured; **none adopted**
- Defaults unchanged; V26 prediction hash unchanged
- Post-2022 outcomes used: none

The first dispersion correction was rejected for helping the two compressed
elections and hurting the two that were not. This records three further
mechanisms tried afterwards, and one correction to the earlier note.

## Correction to the earlier record

The earlier note said renormalisation returns the level that a deviation
expansion moves. That is wrong. Expanding each candidate's deviation around
their own vote-weighted national level is compositional by construction, since
the levels themselves sum to one per region, and the measurement bears it out:
national macro moved only 0.7210 to 0.7228 at the best gain. The dispersion
rescale's problem was never the level. It was that it applied to elections that
did not need it.

## The four mechanisms

| mechanism | regional macro | national macro | worst cell | elections left untouched |
| --- | ---: | ---: | ---: | --- |
| baseline | 2.7122 | **0.7210** | 15.69 | — |
| 1. uniform expansion, indexed on predicted third share | 2.5829 | 0.7228 | — | 2012 |
| 2. symmetric anchoring toward the bloc prior | 3.0501 | 1.4623 | — | none |
| 3. one-sided downward anchoring in hostile regions | 2.8040 | 0.9762 | 14.45 | none |
| 4. historical-maximum bound, x1.30 headroom | 2.5738 | 0.8587 | 13.72 | **2002, 2012, 2022** |
| 4b. the same, level preserved within the candidate | **2.5310** | 0.8068 | 14.05 | **2002, 2012** |

### 2. Anchoring toward the prior fails outright

Pulling predictions back toward the normalised `recent_bloc_base` degrades every
election at every weight tested, monotonically. The 2016 bloc base is stale and
the model is right to move away from it on average; in TK 2017 it moves too far,
but that is a local error inside a globally correct behaviour.

### 3. One-sided downward anchoring helps only 2007

Restricting the pull to candidates predicted above their own bloc prior in
regions hostile to them improves 2007 monotonically - 4.27 to 3.89 - and
degrades everything else. At the strongest weight a winner is lost.

### 4. The historical bound is the first mechanism that scopes itself

Capping each candidate at the maximum share their bloc has ever taken in that
region across prior presidential elections, with headroom, binds only where the
model exceeds precedent. That turns out to be 2007 and 2017 and nowhere else:
2002, 2012 and 2022 come through byte-identical or nearly so.

This is the property the first three lacked. Every earlier mechanism needed an
index to decide where to act, and every index either failed to separate the
elections or separated them by fitting. The bound needs no index because the
condition is the correction.

Holding the candidate's own level - returning the released mass to their other
regions rather than to rivals - gives the best regional figure measured, 2.5310,
and the worst cell falls from 15.69 to 14.05.

## Why none is adopted

**All four degrade the national metric.** 0.7210 goes to 0.7228, 1.4623, 0.9762
and 0.8068. Four mechanisms with different shapes, different indices and
different scopes all move it the same way.

That is now a result rather than four coincidences. V26's national calibration
sits at a local optimum - V25 was 0.9896 and V26 reached 0.7210 - and every
regional intervention tried pushes it back toward 0.8 to 1.1. Regional shape and
national level are in tension in this model, and the shipped configuration is
near the level optimum.

**And the bound's headroom is fitted.** The x1.30 multiplier was chosen from a
sweep on the panel. Without it the bound is too tight - 2007 worsens to 5.16 -
and at x1.15 it disturbs 2012. The mechanism buys self-scoping and pays for it
with a constant selected on the same five outcomes it is measured against.

## What is worth keeping

Mechanism 4b is the best correction found and the only one that leaves the
calibrated elections alone. If regional error or worst-cell error ever becomes
the metric that matters more than national level, it is where to start, and the
headroom should be pre-registered rather than swept.

The broader finding is the tension itself. Across seven attempts now - a
dispersion rescale, personal-history caps in three forms, absolute core
erosion, prior anchoring, and a historical bound - every correction that
improves regional shape costs national level. The panel does not contain the
information to improve both, and the shipped configuration is not obviously
improvable without more elections.
