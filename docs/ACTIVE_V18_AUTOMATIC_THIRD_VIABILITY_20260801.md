# Active V18: Automatic Third-Candidate Viability

Date: 2026-08-01

## Decision

V18 is the active successor to V17 and the V10 automation lineage. It replaces
only the `viability` field of matched third-candidate profiles with a strictly
prior election-derived value. It does not claim a performance improvement.

The following fields remain separately sourced and manual:

- centrist appeal;
- anti-major-party appeal;
- regional-base overlap;
- major-candidate third-pressure controls.

The immediate rollback checkpoint is
`backups/model_checkpoints/20260801_pre_v18/`.

## Automatic evidence

The compiler separates two election channels before constructing viability:

1. Direct party preference: National Assembly PR, metropolitan-council PR, and
   local-council PR results.
2. Local organization: National Assembly district, metropolitan-council
   district, and local-council district results.

These are combined with strictly prior presidential stature, won-office
history, and the existing Assembly role vector. The target election result is
never read. Three rows are replaced automatically:

| Election | Candidate | Manual viability | Automatic viability |
| --- | --- | ---: | ---: |
| 2002 | Kwon Young-ghil | 0.2200 | 0.200286 |
| 2007 | Lee Hoi-chang | 0.6500 | 0.660683 |
| 2017 | Ahn Cheol-soo | 0.9000 | 0.898051 |

The 2012 and 2022 profile rows are retained because there is no active third
candidate after the existing contest/withdrawal treatment.

## Promotion gate

V17 baseline:

| Metric | V17 |
| --- | ---: |
| Regional equal-election macro MAE | 3.213269%p |
| National equal-election macro MAE | 1.481455%p |
| Winner accuracy | 0.80 |

V18 result:

| Metric | V18 | Change from V17 |
| --- | ---: | ---: |
| Regional equal-election macro MAE | 3.216549%p | +0.003280%p |
| National equal-election macro MAE | 1.479382%p | -0.002073%p |
| Maximum single-election regression | - | +0.015377%p |
| Winner accuracy | 0.80 | 0.00 |

The automation-equivalence gate was fixed before promotion:

- regional degradation no greater than 0.01%p;
- national MAE must not regress;
- maximum single-election regression no greater than 0.05%p;
- no performance-improvement claim.

The V18 nested prediction file is byte-identical to the prior V14b ablation
prediction file. Manifest and summary files differ only because they record a
different config/output location and promotion state.

## Rejected experiments

| Experiment | Regional MAE | National MAE | Decision |
| --- | ---: | ---: | --- |
| V14 full automatic profile | 3.222412%p | 1.529878%p | reject |
| V15 election-derived character traits | 3.223118%p | 1.508059%p | retain as shadow |
| V16a auto pressure, manual profile | 3.233424%p | 1.507386%p | reject |
| V16b auto pressure and viability | 3.234600%p | 1.504608%p | reject |
| Semantic low candidate-ballot weights | 3.247813%p | 1.547306%p | reject |

The pressure experiments produced nearly symmetric 2017 pressure and missed
camp/regime rejection asymmetry. The semantic-weight experiment showed that
simply reducing all candidate-ballot weights removes useful organization
persistence together with candidate contamination.

## Next automation boundary

Regional party history must be represented with separate latent channels:

- direct party preference magnitude;
- district organization reliability;
- candidate personal excess;
- party-lineage regional identity and persistence.

Pre-2004 Assembly records currently collapse some parties, including the
Liberal Democrats lineage, into the generic third bloc. Party-level source
rows must be restored before this history can replace manual regional-overlap
or pressure controls. Every generated feature must use only events strictly
before the target election date and must report its contributing election IDs.

Issue shock, withdrawal, endorsement, and regime-rejection asymmetry cannot be
identified from election returns alone. Those controls require dated event,
Assembly-speech, or economic evidence and remain lower-feasibility automation
work.

## Canonical artifacts

- `scripts/run_current_presidential_model.py`
- `scripts/run_active_presidential_model_v18.py`
- `data/config/current_presidential_model.json`
- `data/config/active_presidential_model_v18.json`
- `outputs/election_derived_third_candidate_profile_v14/`
- `outputs/election_derived_third_candidate_profile_v14b/`
- `outputs/active_presidential_nested_v18/`

