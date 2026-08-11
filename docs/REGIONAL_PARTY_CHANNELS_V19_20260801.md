# Regional Party Preference and Organization V19 Experiments

Date: 2026-08-01

## Purpose

The experiment tested whether manual regional identity could be replaced or
corroborated by party-level National Assembly and local-election records. It
restored the Liberal Democrats lineage from NEC constituency rows where the
aggregate history had collapsed the party into the generic third bloc.

Restored lineage aliases include Liberal Democrats, United Liberal Democrats,
People First Party, Liberty Forward Party, Advancement Unification Party, and
other explicit Chungcheong regionalist labels in the source data.

## Factual compiler

`presidential_issue_engine/regional_party_channels.py` separates:

- direct party preference: Assembly PR and metropolitan/local council PR;
- district organization: Assembly and metropolitan/local council district;
- candidate personal proxy: executive and presidential ballots.

The compiler produced 868 two-channel events, including 240 rows with explicit
party-lineage evidence. NEC constituency data restored 1996, 2000, 2004, and
2008 party organization that was not recoverable from the normalized bloc
label alone.

## V19 replacement result

Replacing the active reservoir directly reduced generic third-bloc evidence
and therefore changed both magnitude and reliability.

| Metric | V18 | V19 replacement |
| --- | ---: | ---: |
| Regional MAE | 3.216549%p | 3.221768%p |
| National MAE | 1.479382%p | 1.546870%p |
| Maximum election regression | - | +0.062254%p |

This version was rejected. The main failure was excessive reservoir shrinkage
in 2012.

## V19b corroboration result

V19b preserved every V18 `identity_excess` value and used verified party
lineage only to increase evidence reliability, with one predeclared
corroboration gain of 0.25.

| Metric | V18 | V19b |
| --- | ---: | ---: |
| Regional MAE | 3.216549%p | 3.219354%p |
| National MAE | 1.479382%p | 1.470656%p |
| Maximum election regression | - | +0.016993%p |

V19b improves the national diagnostic and 2007 regional MAE, but regional
macro MAE regresses. It was not promoted. Active V20 therefore retains the V18
full-history reservoir.

An initial V19b run appeared to improve both metrics, but its evaluator did not
explicitly pass the V18 automatic third profile. That run changed two layers
and is invalid as an ablation. The evaluator was corrected; the canonical V19
candidate and corrected V19b prediction hashes then matched exactly. Only the
corrected values above are authoritative.

## Next design

Party lineage should not be a free vote multiplier. The next design should fit
a latent regional terrain with three separately observed states:

1. party preference magnitude from direct-party ballots;
2. organization persistence from district ballots;
3. candidate personal excess from candidate ballots after subtracting party
   terrain.

The active election sample is too small to tune those state equations by
presidential MAE alone. Assembly and local PR/district pairs should provide the
training objective, with presidential elections reserved for downstream
evaluation.

