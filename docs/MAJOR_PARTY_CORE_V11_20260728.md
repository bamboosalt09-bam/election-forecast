# Major-Party Concrete Support V11 (2026-07-28)

## Decision

Active policy: `active_strict_nested_v11_major_party_core_only`

Concrete support is a durable party-lineage floor, not a generic ideological
camp floor. It is therefore restricted to exact pre-normalization lineages:

- `국민의힘`;
- `더불어민주당`.

Progressive, third-lane, regionalist, reform, independent, and minor
conservative or liberal parties receive zero concrete mass. A minor party does
not become eligible merely because `normalize_bloc` maps it to a broad
conservative or liberal camp.

## Previous structural error

The prior history builder normalized party labels before estimating the lower
tail. This allowed multiple parties in one broad camp to contribute to one
durable core estimate. The candidate camp allocator could then share that core
with an ideologically similar non-major candidate.

In the scored folds, the visible consequence was concentrated in 2017. Ahn
Cheol-soo received mean effective concrete mass `0.0381` despite representing
the third lane. Lee Hoi-chang in 2007 already had zero core because his active
bloc was independent.

## V11 mechanism

1. Before broad-bloc normalization, each historical row records
   `major_party_vote_share` only when the raw lineage is exactly one of the two
   eligible major parties.
2. Broad bloc vote share remains available for regional terrain, recent base,
   volatility, and critical-support estimation.
3. The lower-tail concrete estimate uses only `major_party_vote_share`.
4. Each candidate carries `major_party_core_eligible` from the raw candidate
   lineage before normalization.
5. Ineligible candidates have `durable_core_raw`, `direct_party_core_raw`, and
   `candidate_ballot_core_raw` forced to zero.
6. Their historical stable lower-tail support is retained as critical support.
   No electoral mass is discarded merely because it is not concrete.
7. Camp-level core claims are renormalized only among eligible major-party
   candidates. Critical support may still be distributed by broader political
   alignment.

This keeps the intended distinction:

- concrete: highly persistent attachment to one of the two governing-scale
  party lineages;
- critical support: stable but issue- and candidate-responsive attachment;
- swing: the remaining flexible electorate.

## Impact audit

| Election | V10 regional MAE | V11 regional MAE | V10 national MAE | V11 national MAE |
|---|---:|---:|---:|---:|
| 2002 | 4.0164 | 4.0163 | 3.3967 | 3.3965 |
| 2007 | 5.0486 | 5.0486 | 2.7083 | 2.7083 |
| 2012 | 2.7513 | 2.7513 | 1.0003 | 1.0003 |
| 2017 | 4.6058 | 4.6078 | 3.2443 | 3.2711 |
| 2022 | 1.5211 | 1.5211 | 0.0018 | 0.0018 |
| Macro | **3.5886** | 3.5890 | **2.0703** | 2.0756 |

Relative changes are `+0.0004%p` regional and `+0.0053%p` national. Winner
accuracy remains `4/5`. This is statistically and practically neutral on five
development folds. Promotion is based on correcting the latent-variable
definition, not on claiming a performance gain.

The only material layer reclassification is 2017 Ahn Cheol-soo:

| Layer | V10 mean | V11 mean |
|---|---:|---:|
| Effective concrete | 0.0381 | 0.0000 |
| Effective critical | 0.0597 | 0.0965 |

The stable mass is preserved but becomes responsive. His national prediction
moves from `23.255%` to `23.168%`; the actual three-candidate normalized share
is `24.751%`.

## Guardrails

- policy config fixes exact eligible lineages and zero non-major core;
- policy validation fails if the restriction is removed or weakened;
- a dedicated test covers a canonical major party, progressive party, third
  lane, and a minor conservative party that normalizes into the same camp;
- the output exposes `major_party_core_eligible` for row-level audit;
- all history remains strictly before each target election;
- 2025 outcomes remain prohibited.

## Verification

- full test suite: `373 passed`;
- strict deep PIT audit: PASS, outcome invariance `215/215`;
- through-2022 selection-boundary audit: PASS;
- realized-slot leakage audit: PASS;
- active input manifest: `42` hashed files, no 2025 path.

## Artifacts

- `data/config/active_presidential_model_v16.json` (renamed compatibility base);
- `outputs/active_presidential_nested_v11/`;
- `outputs/major_party_core_v11_experiment/decision.json`;
- `scripts/evaluate_major_party_core_v11.py`;
- `archives/experiments/major_party_core_v11_20260728/`.
