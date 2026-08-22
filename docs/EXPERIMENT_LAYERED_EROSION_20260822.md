# The veto ignores the electorate layers, and fixing that is blocked upstream

## Status

- Date: 2026-08-22
- Status: measured; **not adopted**. `floor_erosion_mode` defaults to
  `proportional` and `core_erosion_resistance` to 1.0, both preserving shipped
  behaviour.
- V26 prediction hash unchanged

## The design inconsistency is real

`contest_regime` separates the electorate: concrete support has a floor, and
critical and swing respond at declared elasticities of 0.75 and 1.25.

`strong_incumbent_veto` does not. It erodes the core floor and then applies one
rate to everything above it:

    eroded_floor = base_runner_floor * (1.0 - rupture_floor_activation)
    flexible     = runner_prediction - effective_floor
    transfer     = rate * flexible

Concrete, critical and swing come out at the same rate. In 대구 2017 that let
the veto take **0.0527 from a candidate whose non-core mass was 0.0083**, and
the prediction lands at 0.4274 beneath his own concrete core of 0.4584.

## What was measured

A `layered` mode draws swing at 1.25 and critical at 0.75, with a
`core_erosion_resistance` controlling how much of the eroded core remains
available. At 1.0 the core contributes as before; at 0.0 concrete is protected
outright.

| core resistance | regional macro | national macro | 2017 regional |
| --- | ---: | ---: | ---: |
| shipped, no layers | 2.7122 | 0.7210 | 3.025 |
| **1.0 — elasticities only** | 2.7254 | **0.7061** | 3.043 |
| 0.8 | 2.7499 | 0.7611 | 3.165 |
| 0.5 | 2.7936 | 0.8882 | 3.384 |
| 0.0 — concrete protected | 2.8930 | 1.1001 | 3.881 |

**Monotone.** Every increment of core protection is worse on both metrics, and
there is no interior optimum.

Applying the elasticities while leaving the erosion alone is the one variant
that **improves the national metric** - 0.7210 to 0.7061 - at a regional cost of
0.013. It is the only change measured in this whole line of work that moves the
national figure the right way.

## Why protecting concrete does not help

Because the prediction is already wrong before the veto runs.

The veto takes 0.0527 from 홍준표 in 대구. Blocking all of it leaves him at
0.4801 against a realised 0.5525 - still **7 points short**. The veto is not
what put him below his core; it is what took him from slightly above it to
slightly below.

Upstream, the regional identity shift applied to him in 대구 is **+0.0000**
against a needed +0.2568, and in 경북 +0.0000 against +0.2386. Regionalism is
not reaching the strongholds at all, and a veto that draws less cannot return
support that was never given.

The second obstacle is that the veto's **total** transfer is load-bearing for
the national levels - 홍준표 +0.029 and 문재인 +0.273 under the shipped form -
so any change that shrinks the total breaks that calibration, which is what the
monotone national column shows.

## The ordering this implies

Concrete erosion is worth fixing **after** regionalism reaches the strongholds,
not before. Fixed now it only declines to take from a prediction that is
already too low, and it pays for that with the national calibration the current
total is carrying.

The elasticity part is separable and is the more promising half: it costs 0.013
of regional error and returns 0.015 of national error, without touching how much
the core is eroded at all.

## Related

- `docs/DIAGNOSIS_PERSON_HISTORY_AND_MOVABLE_MASS_20260822.md` - the erosion
  rate is 0.866 nationwide in 2017 while regional core depth varies 0.334 to
  0.483
- `docs/EXPERIMENT_PERSON_HISTORY_AND_CORE_20260822.md` - the absolute-erosion
  variant, which fails the same way
