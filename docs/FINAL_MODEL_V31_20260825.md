<!-- active-model-version: v31 -->
# Final presidential model V31 — a dispersion expansion with no floor to hit

V31 is V30 with one change: the terminal dispersion expansion is multiplicative
rather than additive, so no predicted share can reach zero and the per-election
feasibility cap is gone. Everything else — the Ridge stack, the predictors, the
shock structure, the V28 external-model boundary, V27's regional transform, the
forecast-time weighting and the gain — is V30's.

## Why

V29's additive expansion is linear in the deviation and unbounded below, so it
was capped at the factor where some region reaches zero. That cap is defined by
the first region to reach zero, so that region is published at exactly zero
whenever the cap binds. It bound twice: 홍준표's 광주 in 2017 (3.55% in, 1.68%
realised, 0.00% published) and 김문수's 광주 in the 2025 demonstration (2.67%
in, 0.00% out). Renormalisation then moved the displaced mass onto the other
candidates in that region.

The multiplicative form scales the ratio rather than the difference. A positive
input stays positive at any factor, so the constraint holds by construction and
no cap is needed.

## Frozen development metrics

- regional equal-election macro MAE: `2.5007010072077227%p`
- national equal-election macro MAE: `0.7242913678028117%p`
- winner accuracy: `0.8`
- scored rows: `232`
- lowest predicted share: `1.9688%` (was `0.0001%`)
- post-2022 outcomes used: `false`

Against V30's `2.5664447526782004` and `0.7204374174124484`: regional improves,
**national gets worse**. The change was made anyway. A prediction of exactly
zero for a major-party candidate in a metropolitan region is wrong in kind, and
`+0.0039%p` on a post-hoc diagnostic aggregate does not weigh against that.
Both figures were measured before the decision.

## Level conservation

The multiplicative form does not preserve the weighted national level on its
own — measured drift up to `0.465%p`. The regional sums and the candidate levels
are therefore alternated to convergence, which restores the level to `2.3e-13%p`
while keeping every region at 100%. No constant is introduced: the targets are
the input levels and one. Convergence takes 1–17 rounds; failure to converge
raises.

## AI boundary

Unchanged from V28 and re-audited:

- hosted inference API: none
- downloaded model weights at runtime: none
- external neural encoder at runtime: none
- external-model-derived active input: one compact frozen candidate-issue aggregate
- fitted component: scikit-learn Ridge plus deterministic project transforms

## Evidence boundaries

2002–2022 remain a development panel, not an untouched holdout. The 2025
artifact remains a corrected D-1 demonstration; it was regenerated for V31 and
its winner, ranking and national levels are unchanged.

The scored panel is still defined by which candidates cleared roughly 1% of the
actual vote — a declared modelling scope recorded in
`DIAGNOSIS_SCORING_SCOPE_20260824.md`. The headline metric still weights by
`contest_votes`; that is the definition of national vote share rather than a
leak, and the reasoning is in `METRIC_WEIGHTING_20260825.md`.

Two defects in V30's frozen record — a provenance line saying "equal regions for
the first" that its code no longer did, and an audit requiring the active
version to outscore its predecessor — are recorded in
`EXPERIMENT_V31_MULTIPLICATIVE_EXPANSION_20260825.md` rather than rewritten.
