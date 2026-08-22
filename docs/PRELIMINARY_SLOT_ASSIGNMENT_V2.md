# Preliminary candidate slot assignment v2

## Purpose

The old A/B fields are aligned with realized election rank and therefore cannot
be defended as forecast inputs. This experiment assigns candidate roles from a
separate preliminary expected-share model before the main competition mechanics
run.

## Strict information boundary

- Scored elections: 2002, 2007, 2012, 2017, and 2022.
- Warm-up: 1997 only.
- Each target fold trains only on elections earlier than the target.
- 2025 outcomes are absent from fitting, ranking, thresholds, and diagnostics.
- Target-election vote share, vote count, winner, and realized rank are rejected
  by the slot compiler.
- National preliminary shares use immediately prior presidential regional vote
  volume. The 2002 fold uses equal-region weights because the 1997 warm-up table
  has shares but no turnout counts.

## Preliminary model

The fixed Ridge model uses only these six existing candidate and regional
signals:

1. `issue_advantage`
2. `rif`
3. `partisan_prior`
4. `landscape_bloc_alignment`
5. `landscape_centrist`
6. `landscape_inferred_prior`

`slot_A`, `slot_B`, `slotA_prior`, and `slotB_prior` are excluded. The alpha is
fixed at the existing default rather than selected against target outcomes.

The contextual Ridge share is multiplied by `candidate_weight^2` and then
renormalized. `candidate_weight` is the existing point-in-time assembly-derived
combination of public treatment, party context, coalition cohesion, wasted-vote
resistance, and vote-conversion capacity. The square represents two successive
requirements for support: political presence and conversion into a ballot
choice. A 0.08 floor keeps unprofiled minor ballot candidates materially in the
denominator instead of treating missing profile data as zero support. Neither
the exponent nor the floor is selected against a target election result.

## Assignment rule

- A: highest preliminary national expected share.
- B: second-highest preliminary national expected share.
- C: highest remaining active ballot candidate.
- Alpha: any further active candidate.
- Withdrawn candidate: retained as a transfer reservoir and never assigned C.

C is not deleted at a hard threshold. The model reports
`P(preliminary share >= 5%)` as `third_viability`. A 0.50 probability gate only
changes the descriptive role (`major_third` or `minor_third`); it does not alter
the candidate share or denominator.

## Competition regime and weak hierarchy

The assignment also classifies each election as `two_strong_one_weak`,
`two_strong_one_medium`, or `one_strong_two_medium`. This is a descriptive
prior for the Korean presidential field: it permits 2-strong/1-medium and
1-strong/2-medium elections, but does not assume three symmetric frontrunners.

The optional `preliminary_hierarchy_min2` shadow variant uses this information
as weak national regularization after the vote model has produced its regional
predictions. It is not a regional hard cap. Let `c_model` be the model's
national C share and `c_prelim` the preliminary C share:

```text
logit(c_pooled) = 0.75 * logit(c_model) + 0.25 * logit(c_prelim)
```

The pooled national C share has a loose 30% absolute ceiling and may reach 95%
of B. A single national C multiplier is then applied to every region and each
region is renormalized. This preserves the model's regional C pattern and the
regional A:B ratio. It does not force A to beat B in every region and does not
force a medium C candidate to become minor. Prior-election regional vote volume
is used only to aggregate the national shares, not to alter the regularization
strength.

The through-2022 strict nested shadow comparison after adding the 2002
withdrawal sequence is:

| Variant | Regional weighted macro MAE | National candidate MAE |
|---|---:|---:|
| Slot-free roles only | 7.338%p | 6.518%p |
| Slot-free plus weak hierarchy and no direct neutral vote effect | 5.811%p | 4.819%p |
| Continuous preliminary share, no old direct transfer | 5.723%p | 4.534%p |
| Withdrawal target gate, learned replacement after two prior events | 5.973%p | 4.905%p |

## Active promotion

The `slot_free_hierarchy_no_neutral` row is now the active presidential model.
It retains the final `withdrawn_candidate_transfer` addition after Ridge and
before normalization. The realized-rank slot model remains archived only. The
the then-active configuration is now preserved as
`data/config/active_presidential_model_v16.json`; canonical
strict nested outputs are written by `scripts/run_active_presidential_model.py`
to `outputs/active_presidential_nested/`.

For the hierarchy variant, 2007 national C changes from 31.06% to 23.64%, while
2017 changes only from 17.52% to 18.17%. This is the intended behavior: correct
a large three-way overstatement without suppressing a credible medium third
candidate. These settings were inspected against 2002-2022 diagnostics and are
therefore an engineering comparison, not an untouched holdout estimate.

## Neutral-context direct-effect ablation

The 2022 fold exposed a semantic mismatch in the direct neutral-context layer.
The Ridge and regional normalization predicted A:B at 54.36:45.64, but adding
`0.60 * assembly_neutral_issue_signal` moved A by about -7.69 percentage points
and reversed the field. Neutral mentions describe salience and public attention;
they do not, without directional evidence, identify candidate support.

The strict fold configs already set the neutral-context scale to zero for
2002-2017. Disabling only its direct vote adjustment therefore changes those
four elections by exactly 0.000 percentage points. It changes 2022 from
45.66:54.34 to 53.35:46.65 and reduces its national candidate MAE from 4.723%p
to 2.967%p. Neutral context may remain as issue-salience metadata, but should not
receive a signed direct candidate-vote effect.

During this ablation, the shadow evaluator was also found to import the engine
under a different module name from the fold-configuration context. That caused
the final through-2022 layer settings to be applied to every historical fold.
The imports are now unified and the evaluator fails immediately if the engine
objects diverge. All figures in this document use the corrected evaluator.

## Withdrawn-candidate vote reservoir

Final-ballot C and a competitive candidate who withdrew before election day
must not be treated as the same object. The preliminary generator now restores
the latent candidate from point-in-time political-landscape and withdrawal
records, estimates the candidate in the original A/B/C field, and only then
redistributes that expected share.

For each target, the latent candidate weight is defined so that its square is
the existing two-stage attention-times-conversion structure. Target fractions
are `transfer_rate * voter_compliance`; source mass not transferred to A or B
is treated as abstention or an unscored minor-vote reservoir. Active valid-vote
shares are renormalized after that mass is removed. Final vote outcomes are not
inputs to any of these operations.

The 2002 sequence has two dated events. Roh Moo-hyun became the unified
candidate on 2002-11-25. Chung Mong-joon remained withdrawn, but his
2002-12-18 withdrawal of campaign cooperation attenuates the previously
expected transfer to Roh. The attenuation uses one common conservative rule:
event strength times voter reach times exponential proximity to election day,
with at most half of the original transfer removable. It does not use the 2002
result.

The resulting point estimates are:

| Election | Pre-event A | Pre-event B | Withdrawn C | Post-event A | Post-event B |
|---|---:|---:|---:|---:|---:|
| 2002 | 37.91% | 37.54% | Chung 24.55% | 57.24% | 42.76% |
| 2012 | 45.39% | 33.28% | Ahn 21.33% | 48.25% | 51.75% |
| 2022 | 46.62% | 42.64% | Ahn 10.74% | 53.50% | 46.50% |

The earlier `min2` condition counted any two scored elections rather than two
prior withdrawal elections. The target-gated version now requires both a
withdrawal event in the forecast target and at least two strictly earlier
withdrawal elections. Consequently, only the 2022 fold activates the learned
withdrawal variables, using 2002 and 2012; 2022 outcomes remain excluded. Its
standardized preliminary-withdrawal-share coefficient is 0.0621.

This is formally estimable but not yet superior. Against the otherwise matched
slot-free hierarchy baseline, 2022 national MAE changes from 0.906%p to
1.338%p, regional MAE from 1.632%p to 2.444%p, and five-election national macro
MAE from 4.819%p to 4.905%p. The two examples also differ materially: 2002 had
an election-eve support reversal while 2012 retained the endorsement. N=2 is
enough for one strongly constrained coefficient, but not enough to estimate
separate event-type effects reliably. The target-gated path is therefore kept
as an auditable experiment rather than promoted into the active forecast.
Regional rows increase measurement detail but do not increase the effective
number of independent withdrawal events beyond two; treating them as a larger
event sample would be pseudo-replication. The 2002 viability and political-axis
values are point-in-time structural priors, while the downstream Ridge
coefficient is learned only from earlier realized elections.

## Uncertainty and limits

The current experimental probability uses Ridge coefficient uncertainty over
4,000 draws. It is not the final vote-share prediction interval and does not
yet add a separate residual shock. The candidate source includes the
standardized A/B/C ballot candidates; the aggregate alpha row is excluded.

The generated assignments remain experimental. They must not be fed back as a
fixed A/B vote premium. The intended production use is competition mechanics,
withdrawal transfer, and continuous third-candidate viability after a separate
performance and leakage audit.

## Reproduction

```powershell
python scripts/build_preliminary_slot_assignments.py
python scripts/evaluate_preliminary_slot_shadow_nested.py
python -m pytest tests/test_preliminary_slots.py -q
```

Outputs are written to `outputs/preliminary_slot_assignment/` and
`outputs/preliminary_slot_shadow_nested/`.
