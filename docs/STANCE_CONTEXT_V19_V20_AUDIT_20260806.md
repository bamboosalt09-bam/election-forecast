# Stance Context V19-V20 Audit (2026-08-06)

## Result

V20 is the best independently audited stance classifier in the current
precision-first experiment, but it still fails the forecast-adoption gate.

| Version | Independent audit | Emissions | Harmful errors | Precision | Harmful-error 95% upper |
|---|---|---:|---:|---:|---:|
| V18 | V10 | 59 | 5 | 91.53% | 17.00% |
| V19 | V11 | 75 | 16 | 78.67% | 30.58% |
| V20 | V12 | 73 | 4 | **94.52%** | **12.10%** |

Relative to V18, V20 improves independent precision by `2.99%p` and reduces
the observed harmful-error rate from `8.47%` to `5.48%`. It does not satisfy
the required zero observed harmful errors or 5% upper bound.

Active forecast V23 remains unchanged. No full target-bearing corpus run and
no rolling forecast integration were performed.

## What changed

V19 added general target and owner rules for local government, surveys,
collective actors, foreign support for Korean policy, and an unnamed-prior-
policy context. Its independent result deteriorated because new samples
contained different ownership frames not covered by those rules.

V20 therefore changed the policy more structurally:

1. Positive direction requires explicit first-person ownership.
2. Government officials' descriptions of their own administration are neutral.
3. Diplomatic reports of foreign support are neutral.
4. Party self-position is neutral.
5. Impersonal evaluation/report predicates and their following context are
   neutral.
6. Conditional projections, generic propositions, and multi-government
   historical scope are neutral.
7. Neutral information remains available through the separate information
   score; only directional vote influence is suppressed.

This trades directional coverage for precision. On the fresh E corpus, V20
emitted `53/5000`; on fresh F it emitted `36/5000`.

## Remaining V20 errors

The four independent V12 errors are:

1. A demonstrative government reference inherited the preceding Lee-Park
   governments but was assigned to the current generic government.
2. `If inflation is not controlled, the government may end as a failure` was
   treated as a current negative evaluation.
3. A committee staff report summarizing opposing confirmation opinions was
   treated as the staff speaker's own negative stance.
4. President Moon's statement that hiring corruption is wrong was treated as
   a negative evaluation of Moon.

These are discourse-reference and quotation-owner failures. Adding more
polarity training will not solve them reliably.

## Evidence discipline

- V10 was used to develop V19 and is not independent evidence for V19.
- V11 was used to develop V20 and is not independent evidence for V20.
- V12 uses previously unseen locked emissions from fresh E and F corpora.
- V20 was frozen before V12 texts were reviewed.
- all selected elections are `pres_2002` through `pres_2022`;
- `post_2022_rows_present: false` and `vote_outcomes_used: false`;
- 2025 outcomes were not used.

V12 lock hashes:

- part A: `cf10e2e009690d43627b26f80dac72ce6a05d8b37791ea4a2fae6019ac86a3d2`
- part B: `f43e8b3c50b332155979658234ce4cf1f17702b28cb72c26a617ae60501a89ab`
- labels: `78bdb5e0928219a2c937c06b158d7a5752a9c02a62121492ec262d856ac8edd9`

Authoritative metrics:

`outputs/assembly_stance/stance_context_strict_owner_v20/locked_audit_v12_metrics.json`

## Next improvement

The next version should not add broad sentiment regular expressions. It should
introduce a separate discourse-target resolver with explicit fields:

- `stance_owner`: speaker, target, quoted actor, survey, public, unknown;
- `government_scope`: current, named historical, multiple historical, local,
  foreign, unknown;
- `assertion_status`: asserted, conditional, reported, question, self-policy;
- `referent_source`: current sentence, previous sentence, following sentence.

Only `speaker + current + asserted` rows should reach polarity classification.
This is the remaining path to materially higher precision without collapsing
all coverage.

## Verification

- targeted stance suite: `61 passed`
- full repository suite: `506 passed in 95.23s`
- active forecast integration: intentionally unchanged
