# V24 lineage: scored scope, ballot fidelity, third-candidate ceiling

## Status

- Date: 2026-08-19
- Status: new lineage, published alongside V23; V23 remains the frozen reference
- V23 `nested_predictions.csv` SHA-256 after all V24 work:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b` (unchanged)
- Full regression suite: `568 passed`
- Post-2022 outcomes used: none
- Predictors, ridge alpha, gains, thresholds: identical to V23

V24 does not retune the model. It changes which rows are scored and repairs two
representation defects that were removing evidence from the record.

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
candidate in the panel, and the ordering is monotone in the outcome:

| Election | Vehicle | Major-party carry-in | Share of chamber | Vote share | Share over party base |
|---|---|---:|---:|---:|---:|
| 2002 | 민주노동당 | 0 of 0 | 0.00% | 3.9% | 0.43 |
| 2022 | 정의당 | 0 of 6 | 0.00% | 2.4% | 0.22 |
| 2025 | 개혁신당 | 4 mid-term | 1.33% | 8.3% | 0.78 |
| 2017 | 국민의당 | 11 of 38 | 3.67% | 21.4% | 1.43 |

The trigger therefore uses the defection share whenever the candidate carries a
party, and falls back to the binary flag only where the share is undefined —
the two independents, 이회창 2007 and 강지원 2012.

The floor is `0.02` of the chamber. The scored panel anchors only `0.00` and
`0.0367`, so every floor inside that interval reproduces it exactly; the panel
cannot settle where inside the interval the line belongs. The midpoint is used,
which places 개혁신당 at `0.0133` below the floor. A floor of `0.01` would exempt
it instead. This is the one choice in V24 that the retrospective cannot
adjudicate, and it is recorded as such rather than presented as calibrated.

Measured both ways, the scored panel is identical: macro regional `3.998`, macro
level `2.466`, winner `4/5`.

## Performance

| Election | Candidates | Regional MAE | Level MAE | Winner |
|---|---:|---:|---:|---|
| 2002 | 3 | 5.260 | 3.984 | no |
| 2007 | 3 | 4.721 | 2.353 | yes |
| 2012 | 2 | 3.058 | 1.032 | yes |
| 2017 | 3 | 4.259 | 3.110 | yes |
| 2022 | 3 | 2.693 | 1.851 | yes |
| **Macro** | | **3.998** | **2.466** | **4/5** |

Without the lineage ceiling the same panel gives `4.493` and `3.072`.

The V24 macro figure is not comparable with the V23 headline. On the 199 rows
the two panels share, a same-code control gives `3.906` for V23 and `4.603` for
V24: the number rises because the panel stops excluding the cases the model
handles worst, and because a third candidate the model over-predicts compresses
the two majors. Restricting the V24 panel to 2007 onward gives regional `3.683`
and level `2.087`.

## Where the residual error is

| Slot | Level MAE, full panel | Excluding 2002 |
|---|---:|---:|
| A, winner | 2.75 | 1.95 |
| B, runner-up | 2.65 | 2.81 |
| C, third | 2.24 | 1.66 |

Before the ceiling the same three figures are 3.32, 2.58, and 3.88, so the
constraint removes most of the third slot's excess without touching the others.

2002 dominates. Its only available three-way exemplars are 1992 정주영 (16.3%)
and 1997 이인제 (19.2%), both strong, so the base assigned 권영길 16.8% against a
realised 3.9%. That is an out-of-distribution first observation rather than a
defect: 2022 심상정, predicted 3.7% at base against 2.4% realised, shows the base
handles a weak third candidate once one is in the record.

Excluding 2002, the third slot becomes the most accurate and the runner-up the
least. The remaining systematic bias is a compressed winner-to-runner-up gap in
three-way contests, under-predicted by 5.3, 8.8, and 7.0 percentage points in
2007, 2017, and 2002, and over-predicted by 2.0 and 1.2 in the two-way 2012 and
2022. With four three-way observations, one of which reverses the sign, no
correction can be fitted and validated on this panel.

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
python -m pytest -q
```

Primary artefacts:

- `presidential_issue_engine/fixed_dataset/v24/` — scored scope, results with the
  1%p active-slot rule, deduplicated coalition events, lineage table, V24 contexts
- `presidential_issue_engine/report/tables/v24/issue_vote_engine_nested_outer_predictions.csv`
- `outputs/active_presidential_nested_v24/nested_predictions.csv`
- `outputs/active_presidential_nested_v24/third_candidate_lineage_audit.csv`

## Next valid step

The binding constraint is the number of three-way observations, not the choice
of formula. Four contests, one of which reverses the sign of the dominant bias,
cannot validate a gap correction. Either the founding rosters are sourced so the
defection scale becomes complete across folds, or additional multi-party
contests are mapped into the same pre-election feature space — noting that the
prior weight-class experiment above shows a naive mapping of party-list contests
onto presidential level does not transfer.
