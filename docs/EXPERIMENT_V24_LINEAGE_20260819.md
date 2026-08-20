# V24 lineage: scored scope, ballot fidelity, third-candidate ceiling

## Status

- Date: 2026-08-19
- Updated: 2026-08-20
- Status: new lineage, published alongside V23; V23 remains the frozen reference
- V23 `nested_predictions.csv` SHA-256 after all V24 work:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b` (unchanged)
- Full regression suite: `592 passed`
- Post-2022 outcomes used: none
- Ridge predictors and alpha: identical to V23
- New manual quantities are limited to two declared hypothesis gains, one
  conceptual 10%p activation threshold, and a theoretical 1%p floor; none was
  selected by metric minimisation or by an election-specific flag
- The prediction-tilted recipient rule was promoted only after the original
  affinity-only rule failed the retrospective winner safety gate. V24 execution
  is outcome-blind, but this revision is retrospectively hypothesis-selected.

V24 does not retune the Ridge model. It changes which rows are scored, repairs
representation defects that were removing evidence from the record, and adds
two narrow outcome-blind residual hypotheses whose constants are declared and
sensitivity-tested rather than optimised.

## Why a new lineage rather than a V23 revision

The scored panel is pinned to a frozen artefact,
`presidential_issue_engine/report/tables/issue_vote_engine_nested_outer_predictions.csv`,
and `evaluate_electorate_layers.prepare_frame` guards reproduction against it at
`1e-10`. Any change to the scored row set necessarily regenerates that artefact
and breaks the guarantee. V24 therefore carries its own overrides under
`presidential_issue_engine/fixed_dataset/v24/` and its own baseline under
`presidential_issue_engine/report/tables/v24/`, leaving every V23 path intact.

## Change 1: declared 1%p scored floor

V23 excluded the 2002 third candidate through a single bespoke row in
`scored_contest_scope.csv` whose note reads "2002 scored contest는 노무현·이회창
양강 기준으로 평가". Combined with `is_active_slot`, the effective inclusion rule
was monotone in realised vote share:

| Third candidate | Vote share | V23 scored |
|---|---:|---|
| 강지원 2012 | 0.17% | no |
| 심상정 2022 | 2.38% | no |
| 권영길 2002 | 3.90% | no |
| 이회창 2007 | 15.08% | yes |
| 안철수 2017 | 21.42% | yes |

A per-election exclusion set with outcome knowledge is an evaluation-scope
decision, not a prediction leak, but it is not defensible as reported. V24
replaces it with one declared rule: a slot enters the scored contest when its
national vote share reaches 1.0%p. The rule is uniform, stated in advance of
reading any single result, and recorded with its own source note.

This is still a realised-share criterion. The honest phrasing for a report is
that prediction uses no outcome and no polling, while the scored denominator is
a uniform 1%p floor. No outcome-free alternative reproduces the set: 이회창 2007
ran without a party and would be excluded by any registration or prior-vote rule
despite taking 15.1%.

Panel effect: 199 rows to 232 rows, and two three-way contests to four.

## Change 2: withdrawn candidates no longer occupy the ballot slot

`build_preliminary_slot_assignments._all_ballot_rows` carried this loop:

```python
for profile in profiles.itertuples(index=False):
    mask = frame["election_id"].eq(profile.election_id) & frame["slot"].eq(profile.source_slot)
    frame.loc[mask, "candidate_name"] = profile.candidate_name
```

Every row of the withdrawal registry has `source_slot = C`, so the withdrawn
candidate overwrote the candidate who actually appeared on the ballot:

| Election | Ballot slot C | Overwritten by |
|---|---|---|
| 2002 | 권영길 (3.90%) | 정몽준 |
| 2012 | 강지원 (0.17%) | 안철수 |
| 2022 | 심상정 (2.38%) | 안철수 |

Those are exactly the three weak third candidates. The same function's docstring
states that "withdrawn candidates are not ballot candidates; transfer reservoirs
remain a separate downstream mechanism", so the code contradicted its own
specification. V24 appends the withdrawn candidate on a separate slot and leaves
the ballot candidate in place; the withdrawal transfer is routed through that
slot so `redistribute_withdrawn_vote_mass` is used unchanged.

## Change 3: slot-keyed duplicates of the withdrawal registry removed

`coalition_events.csv` held two rows, `pres_2012 C -> B` and `pres_2022 C -> A`.
`withdrawn_candidate_transfers.csv` already describes the same two events keyed
by candidate name, and in richer form: 2022 carries both an A and a B target.

The slot-keyed copy is not merely redundant. `_apply_coalition_events` transfers
`issue_advantage` and `rif` from the source slot, and `_excluded_event_slots`
drops that slot from the frame. Because the 2022 slot-C row belongs to 심상정,
the engine was transferring 심상정's issue features to 윤석열 while treating them
as 안철수's, and removing 심상정 from the record entirely. The 2012 row did the
same to 강지원. V24 drops the duplicates; the candidate-keyed registry is
untouched and still supplies the 안철수 transfers.

## Change 4: continuous organisation strength

`_organization_strength` returns one of five constants keyed on the bloc name,
so 개혁신당 (three seats, 3.6% list vote) and 국민의당 (thirty-eight seats, 26.7%)
both receive 0.60. V24 makes the non-major branch continuous in the bloc's share
of the latest strictly earlier party-list contest, relative to the strongest
major bloc in that same contest, with the coefficient anchored so pres_2017
제3지대 reproduces 0.60 exactly.

| Election | Third candidate | V23 | V24 |
|---|---|---:|---:|
| 2002 | 권영길 | 0.550 | 0.262 |
| 2007 | 이회창 | 0.150 | 0.150 |
| 2017 | 안철수 | 0.600 | 0.600 |
| 2022 | 심상정 | 0.550 | 0.223 |

Both scored third candidates are unchanged by construction, so the change is
retrospectively neutral: measured against a same-code control run it moves the
macro regional MAE by `-0.0003%p`.

## Change 5: third-candidate lineage ceiling

Across the V24 panel the third candidate separates on whether the vehicle
descends from a large-scale split of a governing or main opposition party:

| Lineage | Candidates | Vote share |
|---|---|---|
| major split | 이회창 2007, 안철수 2017 | 15.1%, 21.4% |
| self-founded or minor | 권영길 2002, 강지원 2012, 심상정 2022 | 3.9%, 0.2%, 2.4% |

The base stage does not carry the distinction, so a self-founded candidate can
receive a level drawn from the strong third candidates the model has seen. V24
caps such a candidate at its own bloc's `direct_party_recent_base` and
redistributes the excess to the two majors in proportion to their predicted
shares.

The rule introduces no fitted quantity. The ceiling factor is exactly one, and
lineage is a documented pre-election fact recorded in
`fixed_dataset/v24/third_candidate_lineage.csv`. It is one-sided: it can only
lower a third candidate placed above its own party base, and it is inert
otherwise. On the panel it fired on twelve regions of 2002 and left 2007, 2012,
2017, and 2022 bit-identical.

An equivalent gate-based trigger (`third_competitiveness_gate < 0.20`) produces
identical retrospective numbers. Lineage is preferred because it classifies the
2007 independent correctly without a special exemption, and because it is an a
priori fact rather than a derived score.

### Trigger basis: defection share where a party exists

The binary flag cannot rank vehicles inside a class. Cross-term name matching in
`data/assembly_roster.csv` recovers the carry-in for every party-backed third
candidate in the scored panel, and it orders them monotonically with the scored
outcome:

| Election | Vehicle | Major-party carry-in | Share of chamber | Vote share | Share over party base |
|---|---|---:|---:|---:|---:|
| 2002 | 민주노동당 | 0 of 0 | 0.00% | 3.9% | 0.43 |
| 2022 | 정의당 | 0 of 6 | 0.00% | 2.4% | 0.22 |
| 2017 | 국민의당 | 11 of 38 | 3.67% | 21.4% | 1.43 |

The trigger therefore uses the defection share whenever the candidate carries a
party, and falls back to the binary flag only where the share is undefined —
the two independents, 이회창 2007 and 강지원 2012.

The floor is `0.02` of the chamber. The scored panel anchors only `0.00` and
`0.0367`, so every floor strictly inside that interval reproduces the panel
exactly and the retrospective cannot distinguish them. The midpoint of the
interval is used. No forecast-only election enters this choice.

Measured across the interval the scored panel is identical. At the restored-
routing and lineage checkpoint, before the two new residual gains, macro
regional-row MAE is `4.013`, region-weighted macro MAE is `3.534`, macro level
MAE is `2.360`, and winner accuracy is `4/5`.

## Change 6: initial strong incumbent-veto tail

V23's coefficient-free rejection-beneficiary routing is restored; an earlier
V24 wrapper call had silently accepted the generic runner's `False` default.
V24 then adds a narrower tail response for a government party facing an already
decisive model-projected defeat. It activates only when all of the following are
known from the outcome-blind forecast frame:

- the government-burdened major candidate is the structural runner-up;
- the equal-region projected gap to the dominant challenger is at least `0.10`;
- contest-regime activation, certainty, and government-rejection strength are
  all positive.

The `0.10` cutoff is a declared structural hypothesis, not an empirically
validated universal boundary. It is retained here so that the assumption is
auditable rather than hidden inside election-specific tuning.

The transfer rate is

`0.50 * government_rejection_strength * dominance_activation * regime_certainty`.

The gain is a round doubling of the initial `0.25` probe, fixed before the
sensitivity table was evaluated. It is not the argmin of that table.

Only the runner-up's mass above its conservative regime core floor can move.
The third-candidate share is unchanged. The rule requests no result, realised
margin, poll, or post-election field, and `pres_2025` is not used to select the
threshold or gain.

The gate fires in 33 scored regions: all 16 regions in 2007 and all 17 in 2017.
Mean additional transfer is `0.865%p` and `0.202%p`, respectively. It remains
off in 2002, 2012, and 2022. Against the same restored-routing and lineage
baseline, this tail alone changes region-weighted macro MAE from `3.534` to
`3.483` and level MAE from `2.360` to `2.235`.

## Change 7: initial weak same-lane wasted-vote refusal

The ordinary strategic-transfer layer depends on candidate conversion context.
That context can be absent or conservative for a weak third candidate, leaving
some same-camp wasted-vote pressure unexpressed. V24 adds a deliberately narrow
hypothesis after the lineage ceiling:

- only slot C candidates whose pre-election vehicle did not receive major-party
  split mass are eligible;
- `candidate_ballot_recent_base` is preserved as a hard lower bound;
- only `max(prediction - candidate_ballot_recent_base, 0)` is transferable;
- a declared quarter of that reservoir moves to A/B according to the existing
  point-in-time political-landscape affinity, with the existing squared
  affinity weighting;
- major-split candidates in 2007 and 2017 are excluded.

The manual gain is therefore `0.25`, interpreted as the hypothesis "one quarter
of weak-candidate support above its concrete prior ballot base is vulnerable".
It is not fitted. The layer fires in 15 regions of 2002 and 13 regions of 2022;
mean transfers are `1.587%p` and `0.346%p`. It is inert in 2007, 2012, and 2017.
No result, poll, realised margin, or forecast-only election is read by the
transformation.

### Declared-gain sensitivity record

The primary pair (`strong=0.50`, `weak=0.25`) was written down before evaluation.
The surrounding grid is diagnostic only. Region-weighted macro MAE (%p):

| Strong gain / weak gain | 0.00 | 0.10 | 0.25 | 0.50 |
|---|---:|---:|---:|---:|
| 0.00 | 3.534 | 3.460 | 3.350 | 3.194 |
| 0.25 | 3.508 | 3.434 | 3.324 | 3.168 |
| **0.50 (declared)** | 3.483 | 3.408 | **3.298** | 3.143 |
| 1.00 | 3.433 | 3.358 | 3.248 | 3.093 |

The monotone surface is evidence that the declared pair is not an isolated
historical optimum; it is not permission to move to the best corner. Every
combination, election-level metric, and transfer row is stored under
`outputs/v24_structural_residual_hypotheses/`, with a manifest separating
transformation inputs from outcome fields used only for retrospective scoring.

## Change 8: floor recalibration and constitutional-rupture response

The first residual revision still made two retrospective states unreachable:

- in 2017 the burdened candidate's mean `regime_core_floor` was `0.270`, so even
  a transfer rate of one could not reproduce a share below that floor;
- in 2022 the weak C candidate's `candidate_ballot_recent_base` was about
  `0.044`, so the initial refusal layer could not approach a lower share.

This is a semantic defect rather than evidence for a larger election-specific
coefficient. A prior party or candidate level is evidence, not an inviolable
concrete floor under a constitutional rupture or severe same-lane wasted-vote
pressure.

### 8.1 Continuous rupture-only core-floor erosion

The ordinary core floor remains unchanged outside a high-intensity direct
government shock. The erosion activation uses only existing forecast-time
fields:

```text
rupture_activation =
    clip(mega_issue_intensity_response - 1, 0, 1)
  * clip(-direct_mega_score / 0.25, 0, 1)
  * government_negative_share
  * sqrt(government_rejection_breadth)

effective_floor = min(
    regime_core_floor,
    max(0.01, regime_core_floor * (1 - rupture_activation))
)
```

The `0.25` score reference already exists in `contest_regime.py`; it is not a
new fitted value. The theoretical floor is `0.01`. The strong-veto gain is the
declared full rejection-strength response, `1.00`.

The activation is exactly zero in 2007 because its intensity is `1.0`, leaving
the existing floor unchanged. In 2017 activation is `0.866`: the mean floor
moves from `0.270` to `0.0368`, and mean transfer rises to `3.344%p`. No manual
"impeachment election" flag is present. This is a large erosion of the concrete
floor and remains a prospective-validation risk even though the response is
continuous and outcome-blind.

### 8.2 Weak-candidate theoretical floor

For a weak, non-major-split C candidate, the protected floor changes from the
entire prior candidate ballot base to a theoretical `1%p`. Half of the mass
above that floor is declared transferable. Strong split-lineage candidates in
2007 and 2017 remain excluded.

The first declared routing sent 100% of removed mass to the candidate with
non-zero same-lane affinity. It reduced MAE but failed the existing winner
safety gate: 2022 flipped to B and total winner accuracy fell from `4/5` to
`3/5`. This rejected result remains in the experiment record:

| Candidate | Regional weighted MAE | Level MAE | Winner |
|---|---:|---:|---:|
| affinity-only declared candidate | 2.659 | 0.973 | 3/5 |

The follow-up changes no gain. It removes structural zero-probability routing by
using each major candidate's current forecast as the base allocation and tilts
it with the existing squared affinity:

```text
recipient_weight_j = prediction_j * (1 + same_lane_affinity_j)^2
```

This keeps the same-lane candidate favoured without asserting that every weak-C
voter must choose that candidate. The follow-up passes the winner safety gate:

| Candidate | Regional weighted MAE | Level MAE | Winner |
|---|---:|---:|---:|
| prediction-tilted follow-up | **2.770** | **1.076** | **4/5** |

All 48 combinations of strong gain, rupture erosion, weak gain, floor mode, and
recipient mode are retained under `outputs/v24_floor_recalibration_hypotheses/`.
Its manifest records the original declaration, its safety failure, the follow-up
rationale, source hash, and the strict absence of 2025 from transformation and
evaluation rows.

`pres_2025` appears in the lineage table as a forecast-only input row carrying
pre-election facts only — four 국민의힘 incumbents held 개혁신당 affiliation
mid-term in the twenty-first Assembly, a share of `0.0133`. Its realised result
is not read here and must not be used to argue where the floor belongs.

### Scope correction

An earlier revision of this document tabulated the 2025 realised share beside
the scored rows and argued the floor by reference to it. That is stage selection
against a forecast-only election and is forbidden by the project scope rule. The
table above now contains scored rows only, and the floor is justified from the
scored anchors alone. The parameter value did not change; the justification did.

## Performance

| Election | Candidates | Regional row MAE | Region-weighted MAE | Level MAE | Winner |
|---|---:|---:|---:|---:|---|
| 2002 | 3 | 3.377 | 2.658 | 2.240 | no |
| 2007 | 3 | 4.833 | 4.041 | 1.121 | yes |
| 2012 | 2 | 3.058 | 2.559 | 1.032 | yes |
| 2017 | 3 | 3.554 | 3.339 | 0.529 | yes |
| 2022 | 3 | 1.530 | 1.252 | 0.456 | yes |
| **Macro** | | **3.270** | **2.770** | **1.076** | **4/5** |

Before the V24-only veto and lineage extensions, the restored V23 postprocess
gives region-weighted macro MAE `4.090` and level MAE `2.918` on this expanded
panel.

The V24 headline still is not directly comparable with V23 because the scored
panel changed from 199 to 232 rows. On each version's declared panel, the
region-weighted headline moves from V23 `3.368` to V24 `2.770` (`-0.598%p`),
while V24 adds the weak third-candidate cases that V23 excluded. The older
regional-row figure is the equal-election macro of unweighted region rows, not the
region-weighted metric used in V23's `summary.json`.

## Where the residual error is

| Slot | Level MAE, full panel | Excluding 2002 |
|---|---:|---:|
| A, winner | 1.02 | 0.44 |
| B, runner-up | 1.31 | 0.94 |
| C, third | 0.85 | 0.96 |

Before the ceiling the same three figures are 3.32, 2.58, and 3.88. The lineage
ceiling removes the largest C excess; the later floor and rupture layers then
change A/B and weak-C allocation explicitly, as recorded above.

2002 dominates. Its only available three-way exemplars are 1992 정주영 (16.3%)
and 1997 이인제 (19.2%), both strong, so the base assigned 권영길 16.8% against a
realised 3.9%. That is an out-of-distribution first observation rather than a
defect: 2022 심상정, predicted 3.7% at base against 2.4% realised, shows the base
handles a weak third candidate once one is in the record.

The final winner-to-runner gap is under-predicted by about `6.19%p` in 2002,
`1.43%p` in 2007, and `1.08%p` in 2017. It is over-predicted by `2.06%p` in
2012 and only `0.05%p` in 2022. The binding residual is now the 2002
out-of-distribution fold, not a general three-way compression pattern.

## Rejected candidates

Five alternatives were built and measured before the accepted change. All are
recorded because the rejections constrain what a future revision can claim.

### 1. Party-base level anchor

Replace the base level with the bloc's prior direct-party evidence, mixing at
weight `w`:

| w | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|
| Macro level MAE | 4.254 | 4.320 | 5.414 | 6.816 | 8.662 |

Monotone degradation. Rejected.

### 2. Party-base correction of the A/B split

| w | 0.00 | 0.15 | 0.25 | 0.35 |
|---|---:|---:|---:|---:|
| Two-way share MAE | 3.00 | 4.00 | 4.67 | 5.34 |

Party evidence puts 문재인 at 44.1% of the 2017 two-way against a realised 63.1%,
because the 2016 list vote was split by 국민의당. Rejected.

### 3. `direct_party_center` gain

The engine already carries a reliability-scaled pull toward the direct-party
centre, disabled at a default gain of zero.

| Gain | 0.00 | 0.15 | 0.30 |
|---|---:|---:|---:|
| Macro regional | 4.493 | 4.476 | 4.504 |
| Macro level | 3.072 | 3.212 | 3.359 |

Only 2007 improves. Rejected.

### 4. Restoring 1992 to the rolling warmup

`ROLLING_WARMUP_ELECTIONS` contains only `pres_1997`, with no documented reason,
while `WARMUP_ELECTIONS` contains both 1992 and 1997. Adding 1992 doubles the
2002 fold's training base and supplies a second three-way exemplar.

| | Macro regional | Macro level | 권영길 base |
|---|---:|---:|---:|
| 1997 only | 4.493 | 3.072 | 16.8 |
| 1992 + 1997 | 4.528 | 3.069 | 16.7 |

No effect: 정주영 at 16.3% reinforces the same strong-third signal. Rejected.

### 5. Prior weight-class model trained on presidential plus party-list contests

A panel of twenty-four multi-party contests was built — seven presidential, six
Assembly PR, six metropolitan council PR, five local council PR — with region
weights from the latest prior presidential vote volume and strictly chronological
fitting.

| | Presidential level MAE | Ordering |
|---|---:|---|
| Prior weight-class model | 7.18 | 3 of 6 reversed |
| V24 model | 3.07 | winner 4/5 |

The failure is structural rather than a fitting choice. Party-list outcomes
persist from prior party-list outcomes — local council PR prior MAE runs 1.7 to
4.8 — while presidential outcomes decouple. For the third bloc the prior is
almost constant while the outcome is not: 진보정당계 prior evidence spans 8.95 to
11.40 across fourteen contests while its realised share spans 2.57 to 13.00.
Shortening the half-life does not recover the signal; tier classification
accuracy is 65% at a twelve-year half-life, 65% at four years, 65% on the single
latest contest, and 70% on the latest two, against a 57% base rate on twenty-three
observations. Rejected.

The same panel does establish a clean regularity worth recording: the ratio of
presidential to prior party-list share is 0.48 for 진보정당계 and 1.90 for
제3지대, with no overlap in presidential outcomes between the two groups
(진보정당계 at or below 6.2%, 제3지대 at or above 16.6%).

## Continuous split strength

`third_candidate_defection_scale` expresses lineage strength as incumbents who
left a major party for the new vehicle, normalised by chamber size. The ordering
matches outcomes, and it ranks 개혁신당 between 정의당 and 국민의당 where the
binary flag cannot.

Coverage is incomplete. Only the twenty-first Assembly records mid-term
affiliation changes, so the count is repo-derivable for 개혁신당 alone (verified
at four, alongside 새로운미래 at four). `data/assembly_roster.csv` stores the
affiliation held at election time and misses the 2015-2016 departures that
produced 국민의당. Rows before 2020 carry `defection_seats_source = needs_source`
and return missing rather than a guess. The module is diagnostic and is not
wired into the forecast.

One case resists the measure entirely: 이회창 2007 has zero party-level defection
because he ran without a party, while his lineage from 한나라당 leadership is
strong. A seat count cannot express personal lineage.

## Reproduction

```powershell
python scripts\run_active_presidential_model_v24.py
python scripts\evaluate_v24_structural_residual_hypotheses.py
python scripts\evaluate_v24_floor_recalibration_hypotheses.py
python -m pytest -q
```

Primary artefacts:

- `presidential_issue_engine/fixed_dataset/v24/` — scored scope, results with the
  1%p active-slot rule, deduplicated coalition events, lineage table, V24 contexts
- `presidential_issue_engine/report/tables/v24/issue_vote_engine_nested_outer_predictions.csv`
- `outputs/active_presidential_nested_v24/nested_predictions.csv`
- `outputs/active_presidential_nested_v24/third_candidate_lineage_audit.csv`
- `outputs/active_presidential_nested_v24/strong_incumbent_veto_audit.csv`
- `outputs/active_presidential_nested_v24/weak_same_lane_refusal_audit.csv`
- `outputs/v24_structural_residual_hypotheses/` - fixed-grid sensitivity metrics,
  election rows, transfer audit, and input/evaluation manifest
- `outputs/v24_floor_recalibration_hypotheses/` - 48 floor/routing variants,
  candidate-level forecasts, transfer audit, and declaration/follow-up manifest

## Next valid step

The binding constraint is the number of three-way observations, not the choice
of formula. Four contests, one of which reverses the sign of the dominant bias,
cannot validate a gap correction. Either the founding rosters are sourced so the
defection scale becomes complete across folds, or additional multi-party
contests are mapped into the same pre-election feature space — noting that the
prior weight-class experiment above shows a naive mapping of party-list contests
onto presidential level does not transfer.
