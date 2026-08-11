# Issue Interpretation Layer V2 Audit

## Decision

`automatic_issue_interpretation_v2` is the active through-2022 issue seed
policy. It is a semantic correction to candidate attribution, not a new
outcome-tuned multiplier search.

## Problems found in V1

1. Overall issue tone could be copied into every candidate's direction even
   when the sentence evaluated only one person, party, or government.
2. The speaker's party could be confused with the target of praise or
   criticism. An opposition attack could therefore look like an opposition
   penalty instead of evidence about the attacked target.
3. Only the top four candidate-issue rows were retained. Missing rows then
   became zero during cross-candidate centering, which could create artificial
   relative advantages.
4. Incumbent presidents who were not current candidates were not resolved to
   the incumbent-continuity candidate slot.
5. Cached automatic seeds could remain stale after the upstream Assembly
   interpretation overlay changed.

## Active interpretation flow

1. Keep all neutral and directional sentences for unsigned issue salience.
2. Accept candidate polarity only when the ambiguity-gated classifier names an
   explicit `person`, `party`, or `government` target.
3. Resolve current candidates and parties to their pre-election slots.
4. Resolve government targets and dated incumbent-president aliases to the
   incumbent-continuity slot from point-in-time responsibility metadata.
5. Aggregate target evidence by election, issue, and slot.
6. Use speaker bloc only as a reliability modifier:
   same-bloc criticism is strengthened, same-bloc praise is damped,
   cross-bloc criticism is damped, and cross-bloc praise is strengthened.
   This modifier never reverses or creates a sign.
7. Generate a complete candidate-by-issue profile. Direction is
   `tanh(1.5 * target_directional_balance)` and is forced to zero when explicit
   target evidence is absent.
8. Generate mega axes from unsigned salience and evidence only: top two issues
   plus at most one high-evidence political-shock issue.
9. Regenerate the automatic seed and SHA256 manifest at every active model run.
10. For an explicitly attributed political shock with pre-election intensity
    above 1.0, apply a bounded direct log-share shift after the nested model.
    This prevents Ridge from reversing a direction that already means
    candidate advantage or burden.

Candidate-wide public treatment and party-tone variables remain diagnostic.
They no longer create issue-specific direction.

## Data boundary

- Scored and selectable elections: 2002, 2007, 2012, 2017, 2022.
- Rolling warmup: 1997.
- Post-2022 outcomes used: none.
- Direction outcome fields used: none.
- Every generated row keeps `available_date` and is filtered at D-1.
- Manual issue seed remains disabled in the active policy.

Sparse `mega_issue_attribution.csv` coverage is intentional. An election with
no explicit target evidence receives no fabricated zero or directional row.
The coverage audit therefore requires candidate profiles and mega axes for all
scored elections, but permits attribution elections to be a strict subset.

## Verified strict nested result

| Election | Regional vote-weighted MAE | National point MAE |
|---|---:|---:|
| 2002 | 3.679%p | 2.485%p |
| 2007 | 12.430%p | 10.934%p |
| 2012 | 3.388%p | 2.681%p |
| 2017 | 6.789%p | 5.863%p |
| 2022 | 1.625%p | 0.881%p |
| Equal-election macro | **5.582%p** | **4.569%p** |

The previous active snapshot was 5.811%p regional and 4.819%p national. The
numerical gain is small. The primary improvement is that candidate direction
now has a defensible target and cannot be generated from generic discussion
tone or speaker affiliation.

## Verification

- `python -m pytest -q -p no:cacheprovider`: 342 passed.
- `python presidential_issue_engine/audit_point_in_time.py --deep`: PASS,
  215 target rows invariant to target-outcome mutation.
- `python presidential_issue_engine/audit_weight_selection_boundary.py`: PASS,
  70,874 active CSV rows and 2,410 report rows checked.
- Active seed output: 247 candidate-profile rows, 15 mega-axis rows, and 6
  explicit mega-attribution rows.

## Remaining limitations

1. Explicit directional evidence is sparse: most Assembly sentences are used
   for salience, not polarity.
2. An issue with no explicit target direction still contributes through the
   engine's pre-existing unsigned issue-ownership path. It is not represented
   as praise or criticism.
3. 2007 and 2017 remain the largest strict nested errors. The direct shock
   layer reduces 2017 materially but does not explain the remaining candidate
   hierarchy error.
4. Speaker bloc metadata is incomplete for some historical speakers; unknown
   relationships receive a conservative reliability weight.
5. This remains a through-2022 development estimate, not an untouched election
   holdout.
