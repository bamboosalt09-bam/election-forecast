# Regional Accent and Regime Diagnosis (2026-07-28)

## Frozen baseline

- Policy: `active_strict_nested_v9_party_context_cohesion`
- Regional contest-vote weighted macro MAE: `3.858372%p`
- National candidate macro MAE: `2.476894%p`
- Winner accuracy: `80%`
- 2025 outcomes remain prohibited.

## Party-context defection rule

V9 uses fixed theory constraints, not empirically fitted cutoffs:

`risk = confidence * [0.65 * (1 - normalized support) + 0.35 * fragmentation]`

`released = risk * (0.02 * core + 0.15 * critical)`

The 2% core and 15% critical limits are fractions of each latent mass, not
percentage-point deductions from total candidate support. A separate safety
cap prevents released mass from exceeding 95% of the candidate prediction.

Released mass is currently returned in proportion to the region's
pre-adjustment candidate prediction:

`p_i' = p_i - released_i + p_i * sum(released)`

This is mass-conserving but does not model where critical supporters or swing
voters actually move. It therefore tends to attenuate slopes rather than create
realistic transitions.

## Quantified residual structure

Weighted predicted-deviation versus actual-deviation slope:

| Election | Slope |
|---|---:|
| 2002 | 0.8943 |
| 2007 | 0.7669 |
| 2012 | 1.0589 |
| 2017 | 0.7694 |
| 2022 | 0.9300 |
| Overall | 0.8665 |

Top-two margin diagnostics:

| Election | Predicted | Actual | Predicted minus actual |
|---|---:|---:|---:|
| 2002 | -4.26%p | +2.43%p | -6.69%p |
| 2007 | 12.18%p | 25.05%p | -12.87%p |
| 2012 | 1.00%p | 3.55%p | -2.54%p |
| 2017 | 11.57%p | 19.70%p | -8.14%p |
| 2022 | 0.54%p | 0.76%p | -0.22%p |

Contest-vote weighted regional standard-deviation ratios (`predicted/actual`)
show under-concentration in most elections. The sharpest examples are 2007 Lee
Hoi-chang `0.510`, 2017 Ahn Cheol-soo `0.612`, 2007 Chung Dong-young `0.732`,
and 2017 Hong Joon-pyo `0.766`. 2012 is the exception at `1.091`, so a global
regional multiplier is not defensible.

## Structural causes

1. Regional history is primarily collapsed to broad conservative, liberal,
   progressive, and third-bloc levels. Candidate-specific affinity exists, but
   finer regional lane composition is not used in the competitive pool.
2. Candidate regional signals are candidate-centered and region-centered
   scalars. They preserve national totals but compress province-specific
   ideological accents.
3. V9 terrain anchoring divides the terrain gain by mega-issue intensity. In
   2017 this reduces the gain from `0.25` to `0.125`, although a shock can
   simultaneously mobilize core supporters and move critical/swing voters.
4. The contest-regime response preserves a conservative core but applies one
   symmetric log shift to all remaining dominant/runner-up votes. Critical and
   swing transitions are not distinguished.
5. Cumulative rejection is reliability-discounted when compiled and then
   passes through another reliability gate in regime activation. This can
   double-attenuate a well-attributed government-rejection signal; 2007 regime
   activation is only `0.387`.

## V10 experiment constraints

- Preserve the conservative core calculation and never increase core mass.
- Derive finer regional accents only from elections strictly before each target.
- Decompose regional affinity into conservative, liberal, progressive,
  centrist/third-lane, regionalist, recent-trend, and volatility evidence.
- Apply the new accent only to the non-core competitive pool.
- Separate critical-support and swing response under regime judgment.
- Avoid target-election-specific constants and keep one policy for all folds.
- Compare against v9 using strict nested regional and national metrics; retain
  v9 if the general policy does not improve the aggregate evidence.

## Implemented regional composition

V10 keeps the broad camp prior, but no longer treats every liberal,
conservative, or third-lane region as internally identical. For every target
fold it reads only elections dated strictly before the target and summarizes
direct party ballots along six axes:

- conservative;
- liberal;
- progressive;
- centrist/third-lane;
- regionalist;
- reform/anti-establishment.

Each region-axis summary contains a time-weighted share, latest-minus-history
trend, weighted mean absolute volatility, and effective-sample reliability.
Candidate matching is `65%` official bloc plus `35%` normalized political-
landscape profile. The resulting score is centered once by candidate across
regions and once by region across candidates. It therefore changes regional
shape but cannot create a free national vote bonus.

The regional log shift is:

`gain * clipped(signal / 0.10) * reliability * (1 - 0.5 * volatility) * noncore_mobility`

where `noncore_mobility = clip(1 - core / baseline_prediction, 0, 1)`. The
fold gain is `min(0.30 * mean_reliability, 0.20)`. It ranges from `0.100` in
2002 to the `0.200` cap in 2022. Observed absolute row log shifts remain below
`0.095`.

This is deliberately conservative about concrete support:

- the new layer never raises estimated core mass;
- high-core candidate-regions receive a smaller shift;
- volatile or weakly supported axes are discounted;
- the signal is zero-sum within each region and candidate shape;
- no target-specific candidate or election constant exists.

## Implemented contest transition

The contest-regime stage now preserves the conservative core floor and splits
the remaining dominant/runner-up pool into critical support and swing support.
The universal response uses critical elasticity `0.75`, swing elasticity
`1.25`, and a swing log-shift cap of `0.50`. Third-candidate share is preserved
by this stage.

Cumulative rejection already contains source reliability. V9 multiplied that
signal by a second reliability gate; V10 removes the duplicate discount while
still requiring the minimum reliability threshold. The resulting activation
is `0.639` in 2007 and `1.000` in 2017, and zero in 2002, 2012, and 2022.

The public function retains its historical symmetric defaults. V10 passes the
new asymmetric elasticities explicitly so older v5/v6 experiments remain
reproducible.

## Reproducible ablation

The assignment seed, outer fit, and ablation are all regenerated under the
same strict undated-input policy. A prior mismatch was traced to regenerating
the preliminary slot assignment outside that policy. The evaluator now fails
if the active full stack and full-ablation predictions differ. Current maximum
row difference is `1.11e-16`.

| Regional accent | Modern regime transition | Regional MAE | National MAE |
|:---:|:---:|---:|---:|
| Off | Off | 3.8659%p | 2.5041%p |
| Off | On | 3.6215%p | 2.1270%p |
| On | Off | 3.8339%p | 2.4462%p |
| On | On | **3.5886%p** | **2.0703%p** |

The regional accent independently improves regional MAE by `0.0320%p` and
national MAE by `0.0578%p` with the legacy regime route. With the modern regime
route active, its improvements are `0.0328%p` and `0.0567%p`. The regime
transition is the larger contributor, but neither result depends on the other.

## V9 to V10 result

| Election | Regional V9 | Regional V10 | National V9 | National V10 |
|---|---:|---:|---:|---:|
| 2002 | 4.0173 | 4.0164 | 3.3455 | 3.3967 |
| 2007 | 6.4407 | **5.0486** | 4.4472 | **2.7083** |
| 2012 | **2.6468** | 2.7513 | 1.2717 | **1.0003** |
| 2017 | **4.6042** | 4.6058 | **3.2122** | 3.2443 |
| 2022 | 1.5828 | **1.5211** | 0.1079 | **0.0018** |
| Macro | 3.8584 | **3.5886** | 2.4769 | **2.0703** |

The equal-share deviation slope rises from `0.8665` to `0.8933`, reducing but
not eliminating central regression. It improves most in 2007 (`0.7669` to
`0.8148`) and 2022 (`0.9300` to `0.9715`). 2012 becomes mildly more
over-dispersed (`1.0589` to `1.0786`), and 2017 remains compressed (`0.7738`).

V10 is promoted because both new components have independent aggregate gains,
the regional error improves in four of five elections except 2012, and the
national error improves in three of five elections. The promotion is not a
claim of untouched holdout performance: these fixed policy choices were
developed while the through-2022 outcomes were visible.

## Remaining error after V10

- 2002 still predicts the wrong winner and has sparse early direct-party
  history.
- 2007 improves materially but still understates Lee Myung-bak by `3.626%p`
  and overstates Chung Dong-young by `4.062%p`.
- 2012's national margin improves while regional MAE worsens, consistent with
  a Chungcheong realignment error rather than insufficient global polarization.
- 2017 remains the clearest unresolved case: Moon Jae-in is understated by
  `3.371%p`, Hong Joon-pyo is overstated by `4.866%p`, and regional compression
  remains strong.
- Released party-context mass is still redistributed proportionally to the
  pre-adjustment regional prediction. Destination choice between same-lane and
  cross-lane alternatives remains unresolved.
- National and regional reported MAEs use observed contest-vote aggregation
  weights and are post-election diagnostics, not deployable forecast weights.

## Verification and artifacts

- full tests: `371 passed`;
- strict deep PIT audit: PASS, outcome invariance `215/215`;
- through-2022 selection-boundary audit: PASS;
- realized-slot leakage audit: PASS;
- input manifest: `42` hashed files, no 2025 path;
- active output: `outputs/active_presidential_nested_v10/`;
- ablation: `outputs/regional_accent_regime_v10_ablation/`;
- frozen archive: `archives/experiments/regional_accent_regime_v10_20260728/`.
