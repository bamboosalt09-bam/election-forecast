# Automatic candidate context v2

Date: 2026-07-28

## Purpose

This experiment replaces candidate-specific third-candidate stature constants,
separates forecast rank from political role, and routes cumulative incumbent
rejection to the major opposition beneficiary without an extra fitted gain.
Active v16 is unchanged.

## New automatic layers

### Political role

`rank_slot` preserves the preliminary expected-share rank. `assigned_slot`
uses the two major-party lineages for A/B and the strongest active non-major
candidate for C. The role assignment uses no target-election vote result.

### Third-candidate stature

`data/raw/third_candidate_profile.csv` is forbidden. Viability is the equal
mean of five level-rank bridges:

1. serious-contender treatment
2. legitimacy treatment
3. organization strength
4. party-elite support
5. coalition stability, defined as one minus fragmentation

Centrist and anti-major-party character come from Assembly-derived political
landscape and treatment signals. There are no election-specific viability
constants. The automatic profiles are:

| Election | Candidate | Viability | Centrist | Anti-major | Confidence |
|---|---|---:|---:|---:|---:|
| 2002 | Kwon Young-ghil | 0.322 | 0.167 | 0.322 | 0.468 |
| 2007 | Lee Hoi-chang | 0.317 | 0.319 | 0.501 | 0.606 |
| 2017 | Ahn Cheol-soo | 0.473 | 0.674 | 0.319 | 0.690 |

### Incumbent-rejection beneficiary routing

The routing changes only A/B flexible support and preserves C. The transfer is:

`cumulative rejection advantage * rejection activation * regime certainty * runner flexible mass`

The cumulative rejection advantage already includes source reliability. No
new election-specific gain or target outcome is used. The layer activates only
for 2007 and 2017 in the current evidence.

## Factorial ablation

| Third profile | Political role | Rejection routing | Regional MAE | National MAE |
|---|---|---|---:|---:|
| manual | rank only | off | 4.279%p | 3.117%p |
| manual | role aware | off | 3.544%p | 2.277%p |
| automatic | rank only | off | 3.464%p | 2.020%p |
| automatic | role aware | off | 3.464%p | 2.020%p |
| automatic | role aware | on | **3.335%p** | **1.781%p** |

Automatic stature already restores the expected ordering in all scored
elections, so role awareness is numerically neutral in that branch. It remains
a required structural guard against a future non-major candidate ranking
second in the preliminary forecast.

## Final v2 national predictions

| Election | A prediction | B prediction | C prediction | National MAE |
|---|---:|---:|---:|---:|
| 2002 | 48.075% | 51.925% | - | 3.142%p |
| 2007 | 51.997% | 29.861% | 18.142% | 1.429%p |
| 2012 | 50.618% | 49.382% | - | 1.155%p |
| 2017 | 44.052% | 32.117% | 23.832% | 2.896%p |
| 2022 | 50.096% | 49.904% | - | 0.284%p |

## Leakage and lineage status

- target excluded from every outer fit: yes
- old realized-slot predictors: absent
- 2025 paths: absent
- manual candidate issue profile reads: zero
- manual mega attribution reads: zero
- manual third-candidate profile reads: zero
- new-layer outcome fields: none

The historical candidate set remains retrospective, and all numeric policy
definitions were developed on the through-2022 sample. This is not an untouched
holdout result.

## Verification

- Full test suite: `407 passed`.
- All candidate-v2 folds exclude the target from fit.
- Realized-slot predictors and neutral direct vote adjustment remain disabled.
- Consecutive context builds are byte-identical.
- The current context hashes match all automatic-profile ablation manifests.
- Active v16 remains `3.381670%p` regional MAE and `1.841654%p` national MAE;
  its config and output were not replaced.

## Remaining manual inputs

The model is not fully automatic. In particular it still reads:

- `third_candidate_pressure.csv`
- `candidate_regional_base.csv`
- `chungcheong_identity_alignment.csv`
- `mega_issue_intensity.csv` and `mega_issue_taxonomy.csv`
- `election_generation_weights.csv`
- withdrawal-event transfer rates

The authoritative inventory is
`outputs/manual_weight_lineage_audit_v2/input_lineage.csv`. Automatic
third-candidate pressure was subsequently tested in v3 and rejected because it
regressed 2002 and 2017. The next replacement priority is candidate regional
base, using strictly prior party ballots and dated office or constituency
history.

The inventory currently records 8 manual or curated lineages still read and 49
fixed numeric parameters. The count is an audit boundary, not 49 independent
fitted degrees of freedom; it includes caps, thresholds, half-lives, and gains.

## Decision

Do not promote yet. The result is slightly better than active v16, but it uses
the same five scored presidential elections and several manual lineages remain.
Freeze this result as the candidate-automation baseline and replace the
remaining inputs one at a time.
