# Current Model Performance (2026-07-28)

> Superseded historical snapshot of V16. The authoritative current model is
> V27; see `docs/FINAL_MODEL_V27_20260822.md` and
> `data/config/current_presidential_model.json`.

## Active policy

`active_strict_nested_v16_regional_identity`

- 1997 is rolling warmup only.
- 2002, 2007, 2012, 2017, and 2022 are scored development folds.
- Every target election is excluded from its own Ridge fit.
- Realized A/B/C rank variables are forbidden. Slots come from rolling
  preliminary expected share.
- Every fold uses the same `structural_mega_shock_regime` pipeline.
- 2025 outcomes are prohibited from fitting, tuning, ablation, and comparison.
- Regional and national metrics use observed target-election contest votes and
  are post-election aggregation diagnostics.

## Aggregate metrics

| Metric | V10 | V11 | V12 | V13 | V14 | V15 | V16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regional contest-vote weighted macro MAE | 3.5886%p | 3.5890%p | 3.5795%p | 3.5807%p | 3.5736%p | 3.3953%p | **3.3817%p** |
| National candidate macro MAE | 2.0703%p | 2.0756%p | 2.0751%p | 2.0791%p | 2.0782%p | 1.8483%p | **1.8417%p** |
| Winner accuracy | 80% | 80% | 80% | 80% | 80% | 80% | **80%** |
| Equal-share deviation slope | 0.8933 | 0.8938 | 0.8951 | 0.8952 | 0.8928 | 0.9011 | **0.9006** |
| Prediction rows | 199 | 199 | 199 | 199 | 199 | 199 | 199 |

The slope statistic measures predicted deviation from the equal-share baseline
against actual deviation. One is ideal for dispersion; below one indicates
central regression. V10 reduces, but does not remove, the compression.

## Election metrics

| Election | Regional weighted MAE | National candidate MAE | Deviation slope | Winner |
|---|---:|---:|---:|:---:|
| 2002 | 3.7484%p | 3.1286%p | 0.8991 | No |
| 2007 | 4.9237%p | 2.5487%p | 0.8216 | Yes |
| 2012 | 2.1992%p | 0.2564%p | 1.0780 | Yes |
| 2017 | 4.5431%p | 3.2492%p | 0.7954 | Yes |
| 2022 | 1.4939%p | 0.0254%p | 0.9795 | Yes |

## National predictions

| Election | Candidate | Predicted | Actual | Error |
|---|---|---:|---:|---:|
| 2002 | Roh Moo-hyun | 48.088% | 51.217% | -3.129%p |
| 2002 | Lee Hoi-chang | 51.912% | 48.783% | +3.129%p |
| 2007 | Lee Myung-bak | 50.520% | 54.140% | -3.621%p |
| 2007 | Chung Dong-young | 32.912% | 29.089% | +3.823%p |
| 2007 | Lee Hoi-chang | 16.568% | 16.771% | -0.203%p |
| 2012 | Park Geun-hye | 51.517% | 51.773% | -0.256%p |
| 2012 | Moon Jae-in | 48.483% | 48.227% | +0.256%p |
| 2017 | Moon Jae-in | 44.309% | 47.476% | -3.167%p |
| 2017 | Hong Joon-pyo | 32.647% | 27.773% | +4.874%p |
| 2017 | Ahn Cheol-soo | 23.044% | 24.751% | -1.707%p |
| 2022 | Yoon Suk-yeol | 50.354% | 50.380% | -0.025%p |
| 2022 | Lee Jae-myung | 49.646% | 49.620% | +0.025%p |

## What V10 changes

The regional terrain now distinguishes conservative, liberal, progressive,
centrist, regionalist, and reform lanes using only prior direct party ballots.
Each lane is reliability- and volatility-adjusted and matched to the
candidate's bloc plus political-landscape profile. Candidate and region
centering prevents a free national support bonus.

Concrete support stays conservative. The regional accent does not change core
mass and its effect is multiplied by the candidate-region's non-core share.
The largest observed absolute regional accent log shift is below `0.095`.

The contest transition preserves a reliability-discounted core floor, moves
critical support with elasticity `0.75`, and moves the swing pool more strongly
with elasticity `1.25`. It also removes duplicate reliability discounting from
the already reliability-weighted cumulative rejection signal.

## Factorial ablation

| Regional accent | Modern regime | Regional MAE | National MAE |
|:---:|:---:|---:|---:|
| Off | Off | 3.8659%p | 2.5041%p |
| Off | On | 3.6215%p | 2.1270%p |
| On | Off | 3.8339%p | 2.4462%p |
| On | On | **3.5886%p** | **2.0703%p** |

Both components improve the aggregate metrics independently. The active and
full-ablation v10 predictions match to `1.11e-16` maximum absolute difference.

## V11 concrete-support correction

V11 restricts concrete support to exact pre-normalization party lineages
`국민의힘` and `더불어민주당`. A minor conservative party is not eligible merely
because it normalizes to the conservative bloc. Progressive, third-lane,
regionalist, reform, and independent candidates also receive zero concrete
mass.

Their historically stable lower-tail support is not discarded. It is retained
as critical support, which can react to issues and contest conditions. Core
sharing inside a broad ideological camp is normalized only among eligible
major-party candidates.

The practical impact is tightly localized. Ahn Cheol-soo's 2017 mean effective
core changes from `0.0381` to `0`, while critical support becomes `0.0965`.
Lee Hoi-chang in 2007 already had zero core as an independent, so that election
is unchanged. Other scored elections contain only the two major lineages and
are effectively unchanged.

Relative to v10, regional macro MAE changes by `+0.0004%p` and national macro
MAE by `+0.0053%p`; winner accuracy is unchanged. This is a theory-correction
promotion, not a performance promotion. The small adverse movement is recorded
rather than hidden.

## V12 strategic lane transfer

V12 does not restore concrete support to minor-party or independent candidates.
Instead, their effective critical-support mass is retained as an ideological
lane reservoir. Its transferable fraction is:

`reservoir * major_party_gravity * (1 - wasted_vote_resistance) * (1 - relative_preliminary_viability) * confidence * lane_clarity`

Recipients are limited to eligible major-party candidates and weighted by
squared ideological affinity. Conservative-to-liberal and liberal-to-
conservative transfer is explicitly forbidden. Transfers conserve the regional
vote-share sum exactly and use no realized outcome fields.

Only 2017 activates in the current scored candidate sets. Mean transfer from
Ahn Cheol-soo is `0.2363%p` by region; all other elections are exact no-ops.
Relative to v11, regional macro MAE improves `0.0095%p`, national macro MAE
improves `0.0005%p`, and winner accuracy stays `80%`.

## V13 same-lane affinity correction

The shared affinity helper previously used a set-intersection condition that
assigned affinity `0.65` to a conservative candidate and a liberal-centrist
candidate. V13 replaces it with the intended exact pair check: `0.65` applies
only to liberal-centrist versus centrist. Cross-camp affinity is zero.

Relative to v12, regional macro MAE changes by `+0.0012%p` and national macro
MAE by `+0.0040%p`; winner accuracy is unchanged. The correction is retained
despite the tiny adverse metric movement because the previous routing violated
the declared same-lane mechanism.

## V14 VIF-gated regional offset

V14 learns stable regional two-camp log-odds offsets using only prior Assembly
PR, metropolitan-council PR, and local-council PR elections. Rolling
non-presidential validation selected the offset model over a regional swing
slope. The fallback requires two prior scored elections and a largest finite
fold VIF above `20`; exact duplicate and constant columns are audited separately.

Only 2012 activates (`gain=0.25`). The other four folds are exact no-ops.
Relative to v13, regional macro MAE improves `0.0072%p`, national macro MAE
improves `0.0009%p`, and pooled Chungcheong MAE improves `0.1710%p`.
Third-candidate mass is preserved and no presidential outcome enters the gate.

An all-fold stress test forced the offset onto every election with an available
prior profile. Regional macro MAE worsened to `3.6407%p`, national macro MAE to
`2.1261%p`, and 2022 regional MAE worsened by `0.2764%p`. The VIF gate is
therefore retained; the offset is not a universal regional multiplier.

## V15 Chungcheong regional-identity reservoir

V15 represents Chungcheong's historical regional-party vote separately from
the conservative/liberal terrain. For every target, it measures strictly prior
third/regional-party vote above the cross-region median. Direct-party ballots
receive full weight and old presidential ballots only `0.35`; time decay,
partial pooling, and volatility reduce sparse or unstable profiles.

The reservoir is not concrete support and is never assigned automatically.
Routing requires dated pre-election evidence: an existing documented candidate
regional base, a regional policy commitment, or a completed party merger. If
no recipient is evidenced, the layer is an exact no-op and the mass remains
critical/swing. Regional candidate shares continue to sum to one.

With the predeclared middle sensitivity (`gain=0.50`, regional cap `8%p`),
regional macro MAE improves from `3.5736` to `3.3953%p`, national macro MAE
from `2.0782` to `1.8483%p`, and Chungcheong macro MAE from `5.2292` to
`3.5944%p`. Regional MAE improves in 2002 (`-0.2504%p`), 2007 (`-0.1249%p`),
and 2012 (`-0.5162%p`); 2017 and 2022 are exact no-ops. Gain sensitivity at
`0.25` and `0.75` preserves the same aggregate direction, so the result is not
unique to one gain value.

## V16 non-Chungcheong regional distinctiveness

V16 extends regional identity outside Chungcheong without copying the
Chungcheong third-bloc reservoir. It estimates each region's total-variation
distance from the cross-region median party distribution using strictly prior
direct-party elections and presidential elections downweighted to `0.35`.
Time decay and partial pooling reduce sparse early profiles.

The layer routes support only where `candidate_regional_base.csv` contains a
dated pre-election candidate-region link. It does not create new candidate
links and excludes Chungcheong, whose v15 predictions remain exactly fixed.
Donor support comes first from the candidate whose camp is least compatible
with the existing PIT regional camp profile.

Gains `0.10`, `0.25`, and `0.50` produced regional MAEs of `3.3817`, `3.3662`,
and `3.3471%p`, and national MAEs of `1.8417`, `1.8336`, and `1.8342%p`.
All election-level regional MAEs were maintained or improved. To avoid choosing
the historical optimum, v16 promotes the smallest passing gain, `0.10`, with a
`4%p` cap. Relative to v15, regional MAE improves `0.0136%p` and national MAE
improves `0.0067%p`; 2007 and 2012 are exact no-ops.

## Interpretation limits

These five folds are not an untouched external holdout. The numeric gains and
policy design were developed while through-2022 outcomes were visible. Strict
nested execution blocks target-fold fitting and target-fold stage selection,
but it cannot retroactively make model-development decisions outcome-blind.

The main residual errors are the 2002 winner reversal, remaining 2007 landslide
compression, and 2017 Moon/Hong level bias. V15 substantially reduces the 2012
regional realignment error but does not eliminate Sejong and Chungbuk
underprediction. The dated alignment rows are forecast inputs and not outcome
fields, but their historical definition was made with the through-2022 period
already visible; they do not create a new untouched holdout.

V15 reduces pooled Daejeon, Sejong, Chungbuk, and Chungnam MAEs to `3.8198`,
`4.4425`, `3.4135`, and `3.0377%p`. See
`docs/CHUNGCHEONG_ERROR_DIAGNOSIS_20260728.md` for the mechanism and audit.

## Verification

- full test suite: `398 passed`;
- strict PIT deep audit: PASS, target-outcome invariance `215/215`;
- through-2022 weight-selection boundary audit: PASS;
- active fold audit: target excluded and realized-slot predictors absent;
- standalone realized-slot leakage audit: PASS;
- active input manifest: `43` SHA-256 records and no 2025 path;
- Assembly match-level PIT audit unavailable in this run:
  `assembly_matches_present=0`.

## Artifacts

- `data/config/active_presidential_model.json`
- `outputs/active_presidential_nested_v16/summary.json`
- `outputs/active_presidential_nested_v16/by_election.csv`
- `outputs/active_presidential_nested_v16/national_predictions.csv`
- `outputs/active_presidential_nested_v16/nested_predictions.csv`
- `outputs/active_presidential_nested_v16/fold_audit.csv`
- `outputs/active_presidential_nested_v16/chungcheong_identity_audit.csv`
- `outputs/active_presidential_nested_v16/regional_identity_audit.csv`
- `outputs/active_presidential_nested_v16/input_manifest.csv`
- `outputs/regional_identity_v16_camp_donor_experiment/decision.json`
- `archives/experiments/regional_identity_v16_20260728/archive_manifest.csv`
- `outputs/chungcheong_identity_v15_experiment/decision.json`
- `outputs/chungcheong_error_audit_v15/summary.json`
- `archives/experiments/chungcheong_identity_v15_20260728/archive_manifest.csv`
- `outputs/vif_gated_regional_offset_v14_experiment/decision.json`
- `outputs/all_fold_regional_offset_v14_experiment/decision.json`
- `outputs/strategic_lane_transfer_v12_experiment/decision.json`
- `outputs/orientation_affinity_fix_v13_experiment/decision.json`
- `outputs/major_party_core_v11_experiment/decision.json`
- `outputs/regional_accent_regime_v10_ablation/summary.json`
- `docs/MAJOR_PARTY_CORE_V11_20260728.md`
- `docs/STRATEGIC_LANE_TRANSFER_V12_20260728.md`
- `docs/SAME_LANE_AFFINITY_V13_20260728.md`
- `docs/REGIONAL_ACCENT_AND_REGIME_DIAGNOSIS_20260728.md`
- `docs/CHUNGCHEONG_ERROR_DIAGNOSIS_20260728.md`
- `archives/experiments/chungcheong_regional_offset_v14_20260728/`
