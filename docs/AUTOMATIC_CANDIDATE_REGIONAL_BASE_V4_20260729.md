# Automatic candidate regional base v4

## Purpose

This experiment tests whether `candidate_regional_base.csv` can be replaced
without reading presidential outcomes or manually assigned regional strengths.
The active model remains `active_strict_nested_v16_regional_identity` throughout
the experiment.

## Automatic signal

`speech_derived_candidate_regional_base.py` uses only information available
before each target election:

- the latest prior direct-party ballot for the candidate's political bloc;
- regional excess over the equal-region mean in that ballot;
- the candidate's speech-derived organization strength and evidence confidence;
- the existing fixed election-type data-quality weights.

The compiler excludes major-party candidates because their organization is
already represented by the party-terrain layer. It also excludes independent
labels because aggregate independent votes cannot be attributed to a specific
presidential candidate. The new layer reads no target vote share or target
regional residual.

## Strict nested ablation

| Variant | Regional weighted MAE | National point MAE | Winner accuracy |
|---|---:|---:|---:|
| Manual candidate v2 baseline | 3.334606%p | 1.781094%p | 80% |
| No candidate regional base | 3.599156%p | 1.795882%p | 80% |
| Automatic non-major-party base | 3.575496%p | 1.771862%p | 80% |

The automatic version is slightly better than no base and slightly better than
the manual baseline on the national diagnostic, but it is materially worse on
the primary regional metric. It is therefore not promoted.

## Election-level diagnosis

Relative to the manual candidate v2 baseline, the automatic version changes
regional weighted MAE by:

| Election | Automatic minus manual |
|---|---:|
| 2002 | +0.122868%p |
| 2007 | +0.463891%p |
| 2012 | +0.000000%p |
| 2017 | -0.043377%p |
| 2022 | +0.661067%p |

The prior-party signal successfully identifies Ahn Cheol-soo's 2017 Honam
organization and improves that election. It also produces an outcome-free
progressive-party organization signal for Kwon Young-ghil in 2002. It cannot,
however, infer personal political bases that are not encoded in the party
ballot history:

- Roh Moo-hyun's PK relationship in 2002;
- Lee Hoi-chang's Chungcheong relationship in 2007;
- Lee Jae-myung's Gyeonggi office history in 2022.

The existing manual CSV mixes those personal-history facts with party
organization and manually chosen strength, depth, and confidence values. A
single automatic source cannot safely replace both concepts.

## Decision

Do not replace the active input. The party-organization component is accepted
as a valid automatic feature generator, but it remains experimental until it
can be combined with a separate, dated candidate office/constituency history
input containing facts rather than fitted strengths. No numeric personal-base
values will be reconstructed from the five presidential outcomes.

Authoritative artifacts:

- `outputs/automatic_candidate_regional_base_v4_ablation/summary.csv`
- `outputs/automatic_candidate_regional_base_v4_ablation/by_election.csv`
- `outputs/automatic_candidate_regional_base_v4_ablation/comparison_by_election.csv`
- `outputs/automatic_candidate_regional_base_v4_ablation/decision.json`
- `outputs/speech_derived_candidate_context_v4/auto_candidate_role/candidate_regional_base.csv`
