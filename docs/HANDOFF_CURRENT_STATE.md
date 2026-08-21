# Current State Handoff

> Authoritative current state: `docs/FINAL_MODEL_V25_20260821.md`. The active
> pointer is `data/config/current_presidential_model.json`; V24 and V23 are immutable
> rollback models. Dated sections below preserve the state and terminology that
> applied when each experiment was recorded and must not be read as current
> active-version pointers.

## Active workspace and boundary

- workspace: `C:\english_folder\poll_project`
- active policy: `active_v25_bounded_runtime_repair_pre_2025`
- rolling warmup: 1992, 1997
- scored/development elections: 2002, 2007, 2012, 2017, 2022
- 2025 outcomes: prohibited from fitting, tuning, ablation, and comparison
- Assembly reprocessing: do not run unless explicitly requested

## Active V25 promotion (2026-08-21)

V25 is the frozen current model. It repairs the V24 runner's accidental bypass
of promoted V23 runtime bindings while preserving V24's accepted
`prediction_tilted` weak-C route and the generic third-candidate
profile/pressure paths used by that route.

- prediction rows: `232`
- regional `contest_votes` weighted, equal-election MAE: `2.7739432320%p`
- national candidate, equal-election MAE: `0.9896196355%p`
- winner accuracy: `4/5`
- canonical prediction SHA-256:
  `218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b`
- V24 rollback SHA-256:
  `edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a`
- V23 rollback SHA-256:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`
- full regression suite: `607 passed`
- post-2022 outcomes used: false

The V25 D-1 prospective run reproduces all 232 frozen historical rows before
emitting 51 target rows and computes no performance metric. See
`docs/PRES_2025_V25_STRICT_PROSPECTIVE_20260821.md`.

The prospective runner now reconstructs the sentence corpus to the same full
speech-row granularity used by the historical Assembly matcher. This recovers
cross-sentence regime context and produces a frequency-only
`political_realignment / 0.75` diagnostic. Six PIT-safe official institutional
proceedings activate the universal class-level
`institutional_crisis / 2.0` gate. Direct shock routing is event-class aligned,
so that crisis intensity scales `regime_change` rather than an unrelated
`withdrawal_event`. Government evidence is retained for the incumbent-burden
compiler but excluded from the person/party-only direct candidate profile.
The weak-C donor uses documented party-origin lane before speech-axis
orientation, classifying Lee Jun-seok as `conservative_centrist`. The corrected
unscored national composition is Kim Moon-soo `37.4256%`, Lee Jae-myung
`56.9775%`, and Lee Jun-seok `5.5969%`. These values must not be compared with
the realised 2025 result for selection or tuning.

## Superseded V24 promotion record (2026-08-20)

V24 was the frozen current model at this dated checkpoint. It retains the V23 numerical base configuration
and adds its ballot-faithful three-way scope, uniform 1%p scored floor,
third-candidate lineage ceiling, strong-incumbent-veto response, and weak
same-lane refusal layer through `scripts/run_active_presidential_model_v24.py`.

- prediction rows: `232`
- regional `contest_votes` weighted, equal-election MAE: `2.7697878398%p`
- national candidate, equal-election MAE: `1.0756680504%p`
- winner accuracy: `4/5`
- canonical prediction SHA-256:
  `edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a`
- V23 rollback SHA-256:
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`
- full regression suite: `595 passed`
- exact temporary-directory reproduction: PASS
- public V24 audit and GitHub boundary audit: PASS
- post-2022 outcomes used: false

The national candidate predictive-interval artifact reports 50%, 80%, 90%,
and 95% historical chronological intervals. Its calibration contains only four
target elections and eleven candidate outcomes, so it is not an untouched
holdout or a future coverage guarantee. The official V24 promotion does not
include the unfinished 2025 prospective runner or its local output.

## Corrected 2025 prospective chain (2026-08-18)

An end-to-end input audit corrected three forecast-only integration defects:
incomplete 22nd-Assembly speaker-party mapping, dropped explicit target
direction, and omission of generated 2025 landscape/third-candidate/automatic
issue-seed files from the final runtime. No V23 formula, coefficient, config,
active pointer, or historical output was changed.

The corrected unscored national output is Kim Moon-soo `40.0938%`, Lee
Jae-myung `37.0492%`, and Lee Jun-seok `22.8571%`, using 2022 valid-vote volume
for regional aggregation. The run manifest retains `outcome_columns_used: []`,
`performance_metrics_computed: false`, and `pres_2025_outcome_present: false`.
Do not compare these values with the realized 2025 result.

The remaining high third-candidate share is now diagnosed as a V23 structural
calibration limitation: the slot-free preliminary Ridge begins near an equal
three-way split, and the existing absolute candidate-scale controls do not
fully offset it. Any correction belongs in a separately measured V24 change.
See `docs/PRES_2025_INTERMEDIATE_CHAIN_CORRECTION_20260818.md`.

## 2025 forecast-only Assembly context (2026-08-10)

The requested 2025 pre-election minutes supplement is complete against the
current official National Assembly hierarchy.

- official meetings discovered/completed: `239/239`
- meeting range: `2025-01-06` to `2025-05-14`
- derived issue rows: `48,588`
- D-1 eligible meetings: `91`; retained but excluded: `148`
- D-1 eligible rows: `14,985`; post-cutoff rows excluded: `33,603`
- combined `pres_2025` input rows included: `537,062`
- duplicate rows: `0`
- 2025 outcome fields used: none
- 2025 metrics computed: false
- boundary audit: PASS
- active V23 historical audit: PASS, outcome invariance `215/215`

Meeting date is not used as availability. The official PDF creation date plus
one full day is the conservative proxy; missing metadata fails closed. The
official site does not expose exact first-publication timestamps, so this
limitation remains explicit. See
`docs/PRES_2025_FORECAST_ONLY_ASSEMBLY_CONTEXT_20260810.md` and
`docs/PRES_2025_DEMO_BOUNDARY_AUDIT.json`.

## Latest stance-classifier audit (2026-08-06)

The precision-first stance experiment advanced through V18 in shadow mode.
V18 improved independent directional precision to `91.5254%`, but still made
`5/59` harmful neutral-to-direction errors; the 95% upper bound is `16.9963%`.
It therefore failed the zero-error / 5%-upper-bound adoption gate.

- active forecast V23: unchanged
- full target-bearing corpus classification: not run
- rolling forecast integration test: not run because classifier quality failed
- 2025 outcomes: not used
- authoritative audit: `docs/STANCE_CONTEXT_V15_V18_AUDIT_20260806.md`
- metrics: `outputs/assembly_stance/stance_context_speaker_scope_v18/locked_audit_v10_metrics.json`
- regression suite: `494 passed in 101.76s`

Do not treat the zero-error V17-on-V8 or V18-on-V9 figures as independent;
those audits were development evidence for the next rule version.

### V19-V20 follow-up

V20 is now the best shadow stance classifier. Independent V12 audit precision
is `94.5205%` with `4/73` harmful errors and a `12.1015%` 95% upper bound.
This improves V18 precision by `2.99%p`, but still fails the zero-error / 5%
upper-bound adoption gate. V20 is not connected to active V23 and the full
target-bearing corpus has not been run.

- record: `docs/STANCE_CONTEXT_V19_V20_AUDIT_20260806.md`
- metrics: `outputs/assembly_stance/stance_context_strict_owner_v20/locked_audit_v12_metrics.json`
- regression suite: `506 passed in 95.23s`

### V20 broad-sample follow-up

V20 was additionally applied to a new 10,000-row analysis slice drawn from a
25,000-row broad corpus spanning the 15th-21st Assemblies, five elections,
three target types, 19 issue groups, and 579 coverage cells. Unlike fresh E/F,
the new selector does not force 80% directional-cue rows.

- new broad 10,000: positive `0`, negative `33`, neutral `9,967`;
- representative half: positive `0`, negative `11`, neutral `4,989`;
- combined with fresh E/F: `1:121:19,878` over 20,000 rows;
- V20 directional coverage falls from `0.89%` on cue-rich E/F to `0.33%` on
  the broad sample and `0.22%` on its representative half;
- remaining emitted errors still include wrong stance-owner/target assignment;
- V20 remains shadow-only and active V23 remains unchanged.

Record: `docs/STANCE_CONTEXT_BROAD_SAMPLE_V20_20260806.md`.

### V21-V25-S independent role-gate follow-up (2026-08-07 to 2026-08-10)

The shadow stance line advanced through V25-S with every version frozen before
its next independent audit. None passed the zero-harmful-error / 5%-upper-bound
quality gate.

| Version | Independent precision | Harmful errors | Upper 95% |
|---|---:|---:|---:|
| V21 | 92.63% | 7/95 | 13.39% |
| V22 | 93.15% | 5/73 | 13.86% |
| V23-S | 86.15% | 9/65 | 22.92% |
| V24-S | 92.39% | 7/92 | 13.82% |
| V25-S | 83.91% | 14/87 | 24.01% |

V24-S and V25-S were each tested on a fresh 40,000-row corpus balanced across
2002-2022. V25-S regressed when its untouched corpus shifted from cue-rich to
ordinary parliamentary language. This demonstrates that cumulative regex
patching does not generalize reliably to owner-target-event roles. Stop that
approach after V25-S; the next experiment should be a separately trained,
abstention-capable owner/target/polarity classifier with grouped validation.

- active forecast V23: unchanged
- full-corpus stance integration: not run
- rolling forecast integration: not run
- 2025 outcomes: not used
- record: `docs/STANCE_CONTEXT_V21_V25S_AUDIT_20260807.md`
- latest metrics:
  `outputs/assembly_stance/stance_context_semantic_role_v25s/locked_audit_v17_metrics.json`
- regression suite: `535 passed in 179.48s`
- active pointer and V23 finalization hashes remained unchanged

### V26-S to V29-S external-model follow-up (2026-08-10)

Cumulative regex patching was replaced by experiments grounded in external
target-stance extraction, Korean NLI, and selective-classification work. A
pinned Apache-2.0 KorNLI RoBERTa model, a pinned Korean NLI sentence encoder,
three-stage target/owner/polarity cascades, veto-only policies, task-specific
heads, and representation consensus were evaluated in shadow mode.

V28-S appeared promising on the V16-V17 pseudo-holdout: the embedding-only
veto retained 68 rows with 97.06% precision and an 8.97% harmful-error upper
bound. Those errors were inspected, so V16-V17 were then treated as development
only. V29-S was frozen and tested on 113 newly locked, previously unaudited
base emissions. It achieved only 84.96% precision with 17 harmful errors and a
21.71% upper bound. The improvement did not generalize to the harder pool.

- V29-S independent emissions: `113`
- correct: `96`
- harmful: `17`
- precision: `84.96%`
- harmful upper 95%: `21.71%`
- active forecast V23: unchanged
- full-corpus stance integration: not run
- rolling forecast integration: not run
- 2025 outcomes: not used
- record: `docs/STANCE_EXTERNAL_MODELS_V26_V29_AUDIT_20260810.md`
- decision: do not promote; fixed embeddings and generic NLI are not a
  sufficient owner-target parser
- regression suite: `538 passed in 184.00s`
- backup: `backups/model_checkpoints/20260810_external_stance_v26_v29_shadow`

The statistics-competition archive is separate. This workspace is the active
open-source presidential forecast engine.

## Active forecast structure

1. Candidate slots are assigned from strict rolling preliminary expected share.
   Realized winner rank is not used.
2. Every outer Ridge fit excludes its target election.
3. Active Ridge predictors are issue advantage, RIF, partisan prior, and three
   political-landscape fields. Realized-slot fields are forbidden.
4. Candidate conversion, regionalism, within-bloc transfer, and prior
   direct-party terrain are applied from strictly prior data.
5. Core, critical-support, and swing masses receive different issue responses.
6. The automatic issue seed uses unsigned Assembly salience plus explicit
   person/party/government attribution. Direct manual issue-seed loading is
   disabled. Whether older precomputed context artifacts contain a material
   manual contribution remains under review; the first removal experiment was
   confounded by a missing Assembly match source and was not promoted.
7. A bounded direct-mega shift handles explicitly attributed shocks.
8. The v4 incumbent-shock response adds:
   - weak governing-camp burden from explicit government-target evidence;
   - resistance from prior direct-party strength and forecast conversion;
   - additional response only for shock intensity above 1.0.
9. The v5 contest-regime gate fixes only a reliability-discounted conservative
   core floor, classifies the contest, and reallocates the dominant/runner
   flexible pool. Third-candidate shares are preserved exactly.
10. The v6 cumulative-rejection signal combines explicit government-target
    negativity, negative issue breadth, and either prior-party erosion or an
    attributed rupture route. It applies to 2007 and 2017 without named-candidate
    or election-specific constants.

11. V8 applies `structural_mega_shock_regime` to every historical fold and to
    future elections. Each component is evidence-gated and becomes a no-op when
    its dated input evidence is absent. Candidate stages remain diagnostics;
    target-specific historical stage selection is no longer active.
12. Strict v8 neutralizes the undated curated issue-importance and regional
    sensitivity tables. Every CSV actually read is hashed in the run manifest.
13. V9 treats party context as within-camp cohesion rather than total public
    support. Weak or fragmented context can release at most 2% of core mass and
    15% of critical-support mass into the regional flexible pool. There is no
    direct party-context vote bonus or penalty.
14. V10 derives conservative, liberal, progressive, centrist, regionalist,
    and reform regional accents from strictly prior direct party ballots. The
    signal is candidate- and region-centered and affects only the non-core
    competitive share. Core mass is never increased by this layer.
15. V10 preserves the contest core floor and applies a smaller regime shift to
    critical support (`0.75`) than to swing support (`1.25`). It also removes a
    duplicate reliability discount from cumulative rejection while retaining
    the minimum evidence gate.
16. V11 reserves concrete support for exact `국민의힘` and `더불어민주당`
    lineages before broad-bloc normalization. Other stable lower-tail support
    becomes critical support, not concrete support or discarded mass. Broad
    ideological similarity cannot transfer major-party core to a minor-party,
    progressive, third-lane, regionalist, reform, or independent candidate.
17. V12 keeps stable nonmajor support as a broad ideological-lane reservoir,
    not candidate concrete support. A bounded amount can move to an aligned
    major-party candidate when preliminary viability is weak and dated
    wasted-vote evidence indicates pressure. The transfer is zero-sum within
    each region and never crosses conservative/liberal camps.
18. V13 fixes the shared orientation-affinity condition used by the older
    within-bloc layer. A liberal-centrist candidate now has affinity `0.65`
    only with a centrist candidate, never with a conservative candidate.
19. V14 learns a regional two-camp log-odds offset only from prior direct-party
    elections. It is a fallback after two scored elections when the largest
    finite fold VIF exceeds 20. Exact duplicate predictors are audited
    separately. Third-candidate mass is preserved. Only 2012 activates in the
    current scored folds.
20. V15 estimates a separate Chungcheong regional-identity reservoir from
    strictly prior regional third-bloc excess. It routes only the evidenced
    fraction using dated candidate regional-base, policy-commitment, or party-
    merger facts. Unrouted mass stays critical/swing; 2017 and 2022 are exact
    no-ops.
21. V16 measures non-Chungcheong regional distinctiveness from strictly prior
    party distributions. It routes only to dated candidate-region links and
    takes support first from camps least compatible with the prior regional
    profile. The active gain is the smallest passing sensitivity, `0.10`;
    Chungcheong is excluded and remains exactly unchanged.

The numeric gains inside the pipeline were historically developed on the
through-2022 sample. V8 uses one future-deployable policy instead of selecting a
different stage for 2007 after observing its error, but it does not turn those
five elections into an untouched historical holdout.

## Current verified performance

Primary active nested diagnostics:

- regional contest-vote weighted equal-election macro MAE: `3.367899%p`
- national candidate equal-election macro MAE: `1.597845%p`
- winner accuracy: `4/5` (`80%`)
- prediction rows: `199`
- regional compositional sum maximum error: `2.22e-16`

| Election | Regional weighted MAE | National candidate MAE |
|---|---:|---:|
| 2002 | 3.9816%p | 3.3220%p |
| 2007 | 4.7596%p | 1.6761%p |
| 2012 | 2.6669%p | 0.1627%p |
| 2017 | 4.0658%p | 2.6526%p |
| 2022 | 1.3656%p | 0.1758%p |

Relative to v9, v10 lowers regional macro MAE by `0.2697%p` and national macro
MAE by `0.4066%p`. A strict 2x2 ablation shows independent aggregate gains
from both the regional accent and the critical/swing regime transition. The
active and full-ablation predictions agree to `1.11e-16`.

V11 is a latent-layer definition correction. Relative to v10, regional MAE is
`+0.0004%p` and national MAE `+0.0053%p`, with winner accuracy unchanged. The
change is almost entirely 2017 Ahn Cheol-soo: effective concrete support falls
from `0.0381` to `0`, while stable support is retained as critical support.

V12 adds destination-aware tactical transfer for that retained nonmajor mass.
Only 2017 activates in the scored candidate sets: Ahn Cheol-soo transfers a
regional mean `0.2363%p` to the aligned major-party candidate. Relative to v11,
regional MAE improves by `0.0095%p` and national MAE by `0.0005%p`; all other
elections are exact no-ops.

V13 removes an unintended conservative-to-liberal-centrist affinity in the
pre-existing within-bloc layer. Relative to v12, regional MAE changes by
`+0.0012%p` and national MAE by `+0.0040%p`. This small adverse movement is
accepted because cross-camp transfer contradicts the model definition.

V14 uses a non-presidential regional offset only as a severe-collinearity
fallback. Relative to v13, regional macro MAE improves `0.0072%p`, national
macro MAE improves `0.0009%p`, and pooled Chungcheong MAE improves `0.1710%p`.
The four stable folds are exact no-ops.

The required all-fold stress test was rejected. Applying the offset wherever a
profile exists worsens regional macro MAE from `3.5736` to `3.6407%p`, national
macro MAE from `2.0782` to `2.1261%p`, and 2022 regional MAE by `0.2764%p`.

V15 fixes the distinct missing Chungcheong layer rather than broadening the
v14 camp offset. Relative to v14, regional macro MAE improves `0.1783%p`,
national macro MAE improves `0.2299%p`, and Chungcheong macro MAE improves
`1.6347%p`. All three activated folds improve and the two unevidenced folds do
not move.

V16 adds only evidence-gated residual regional identity outside Chungcheong.
Relative to v15, regional macro MAE improves `0.0136%p` and national macro MAE
improves `0.0067%p`; 2007 and 2012 are exact no-ops. Larger gains improved the
same development sample further, but were not promoted to avoid retrospective
gain optimization.

The authoritative active-model record is
`docs/ACTIVE_V23_UNIFIED_WITHDRAWAL_GENERATION_20260802.md`.

## Speech-derived issue profile experiment

The first deterministic replacement for the retrospective manual candidate
issue strengths is implemented under
`outputs/speech_derived_issue_context_v1`. It derives association, direction,
and confidence from Assembly issue emphasis, salience, evidence coverage, and
explicit target attribution. A complete CSV-read manifest confirms zero reads
of `data/raw/candidate_issue_profile.csv` and
`data/raw/mega_issue_attribution.csv`.

The strict nested result is `4.279040%p` regional macro MAE, `3.116730%p`
national point MAE, and `80%` winner accuracy. This is better than deleting
the manual lineage without replacement (`4.348935%p`, `3.324693%p`, `60%`),
but worse than active v16. The principal remaining failure is 2017 target
coverage: the transcript overlay identifies conservative-bloc burden but does
not identify the principal opposition beneficiary strongly enough to separate
Moon Jae-in from Ahn Cheol-soo.

The experiment is not promoted. Active v16 remains unchanged. See
`docs/SPEECH_DERIVED_ISSUE_PROFILE_V1_20260728.md` and
`outputs/speech_derived_issue_context_v1/decision.json`.

The 2017 failure was subsequently decomposed. The automatic preliminary rank
assigned Moon to C and Ahn to B, so the hierarchy constrained the major-party
opposition candidate instead of the political third candidate. Restoring only
political-role identity in a temporary counterfactual reduced 2017 national
MAE from `8.138759%p` to `3.941317%p`. The remaining error comes partly from a
manual `third_candidate_profile.csv` row that raises Ahn's viability and partly
from proportional redistribution of Hong's regime-rejection loss to Ahn. See
`docs/SPEECH_DERIVED_2017_FAILURE_AUDIT_20260728.md`. No counterfactual was
promoted and active v16 remains unchanged.

## Automatic candidate context v2

A separate candidate v2 experiment now removes
`data/raw/third_candidate_profile.csv`, separates preliminary rank from
major-party/non-major political role, and adds gain-free cumulative-rejection
beneficiary routing. Its strict nested metrics are `3.334606%p` regional macro
MAE and `1.781094%p` national point MAE. The improvement is stepwise in the
factorial ablation: speech v1 `3.116730%p` national, role-aware manual profile
`2.277242%p`, automatic profile `2.019790%p`, and automatic profile plus routing
`1.781094%p`.

This result is not promoted. It is evaluated on the same through-2022 sample,
and manual third-candidate pressure, regional-base, mega-intensity, generation,
and withdrawal-transfer inputs remain. See
`docs/AUTOMATIC_CANDIDATE_CONTEXT_V2_20260728.md` and
`outputs/speech_candidate_v2_ablation/decision.json`.

## Automatic third-candidate pressure v3

The first follow-up replacement targeted `third_candidate_pressure.csv`.
An outcome-free formula allocates `sqrt(centrist * anti-major)` draw capacity
across major-party source lanes using political-axis affinity. Manual pressure,
no pressure, and automatic pressure were compared under the same strict nested
candidate-v2 pipeline.

The automatic version is not promoted: regional/national MAE are
`3.379018%p`/`1.846097%p`, compared with the manual v2 baseline
`3.334606%p`/`1.781094%p`. It improves 2007 but regresses 2002 and 2017. Removing
pressure entirely reaches `3.365990%p`/`1.811908%p`, showing that dependence on
the manual file is small. The remaining missing structure is a general
candidate-affinity by source-camp-rupture interaction; the 2017 split must not
be retuned from its observed result. See
`docs/AUTOMATIC_THIRD_PRESSURE_V3_20260729.md` and
`outputs/automatic_third_pressure_v3_ablation/decision.json`.

## Automatic candidate regional base v4

The next one-lineage replacement separates prior party organization from
candidate personal regional history. An outcome-free compiler uses the latest
strictly prior direct-party ballot, regional excess, and speech-derived
organization evidence for non-major-party candidates. It reads no manual
`candidate_regional_base.csv` rows in its automatic variants.

The full replacement is not promoted. Its strict nested regional/national MAE
is `3.575496%p`/`1.771862%p`, versus the manual candidate-v2 baseline
`3.334606%p`/`1.781094%p`. The automatic signal improves 2017, where prior
party ballots identify Ahn Cheol-soo's Honam organization, but it cannot infer
Roh Moo-hyun's PK, Lee Hoi-chang's Chungcheong, or Lee Jae-myung's Gyeonggi
personal history. Those omissions cause concentrated regional regression.

The reusable party-organization compiler is retained as an experimental
component. The next defensible replacement requires a dated factual
office/constituency-history input with no fitted strengths. See
`docs/AUTOMATIC_CANDIDATE_REGIONAL_BASE_V4_20260729.md` and
`outputs/automatic_candidate_regional_base_v4_ablation/decision.json`.

## Automatic mega-issue intensity v5

The manual election scalar was next tested against an outcome-free Assembly
compiler. Speech-only intensity reaches `3.778872%p` regional and `2.606802%p`
national MAE in the candidate-v2 strict pipeline. A dated event-class gate that
reads no taxonomy numeric fields improves this to `3.456403%p`/`2.016320%p`,
but the manual baseline remains better at `3.334606%p`/`1.781094%p`.

Neither automatic file is promoted. The experiment did reveal and fix a real
consumer defect: values just above the direct-mega threshold previously
activated a full shift. Activation now ramps continuously above the threshold.
The current active input has its only active row at intensity 2.0, so active
v16 is byte-identical before and after the fix: `3.381670%p` regional,
`1.841654%p` national, 80% winner accuracy, and the same nested-prediction
SHA-256. See `docs/AUTOMATIC_MEGA_ISSUE_INTENSITY_V5_20260729.md`.

## Validation status

- full test suite: `416 passed` on 2026-07-29 after automatic mega-intensity v5
- strict PIT audit: PASS
- through-2022 weight-selection audit: PASS
- active fold audit: target excluded, scored denominator consistent, realized
  slot predictors absent, neutral direct adjustment absent
- candidate v2 fold audit: the same four guards PASS in every fold
- candidate v2 context regeneration: byte-deterministic across consecutive runs;
  all automatic-input hashes match the strict nested ablation manifests
- candidate v2 manual-lineage audit: 3 manual candidate-context inputs replaced,
  8 manual or curated input lineages still read, 49 fixed numeric parameters
  inventoried
- candidate v3 automatic-pressure audit: manual pressure read count zero,
  strict nested ablation completed, not promoted because 2002 and 2017 regress
- candidate v4 automatic-regional-base audit: manual regional-base read count
  zero in automatic variants; party organization is reproducible from prior
  direct-party ballots, but full replacement is rejected because personal
  candidate history is absent
- mega-intensity v5 audit: automatic variants read zero manual intensity rows
  and zero post-2022 paths; taxonomy-gated variant reads categorical event facts
  only; neither automatic intensity is promoted
- direct-mega consumer audit: continuous threshold activation reproduces the
  active v16 nested prediction file byte for byte
- standalone slot-leakage audit: PASS against active v11 fold audit
- active input manifest: 43 files, SHA-256 complete, no 2025 path, and no
  undated curated issue-importance or regional-sensitivity input

## Remaining structural errors

- A quantified central-regression pattern remains: overall equal-share
  deviation slope is `0.9006`; 2007 and 2017 are `0.8216` and `0.7954`.
- 2007 remains the largest regional error. Lee Myung-bak is underpredicted by
  `3.621%p` and Chung Dong-young is overpredicted by `3.823%p` nationally.
- 2017 improves only modestly. Hong Joon-pyo remains overpredicted by
  `4.874%p` and Moon Jae-in underpredicted by `3.167%p` nationally.
- 2002 still predicts the wrong winner. The 1997 warmup and early direct-party
  history remain sparse and politically less comparable.
- 2012 national error is now `0.2564%p`, but Sejong and Chungbuk remain below
  the observed conservative share. Further gain tuning on the same outcomes is
  not justified.
- Party-context released mass is still returned proportionally to the regional
  pre-adjustment prediction; destination choice is not explicitly modeled.
- National metrics use observed contest votes as post-election aggregation
  weights. They are diagnostics, not deployable pre-election national weights.
- Chungcheong is now explicitly represented, but the recipient evidence is
  sparse and historically curated. Future forecasts require dated evidence or
  the layer remains a no-op; this is intentional.

Do not further increase gains by reading the same five outcomes. The next
defensible step is external validation or a predeclared structural rule tested
against non-presidential proportional elections without using presidential
targets for coefficient selection.

## Key artifacts

- `data/config/active_presidential_model.json`
- `outputs/active_presidential_nested_v16/summary.json`
- `outputs/active_presidential_nested_v16/nested_predictions.csv`
- `outputs/active_presidential_nested_v16/national_predictions.csv`
- `outputs/active_presidential_nested_v16/stage_selection_audit.csv`
- `outputs/active_presidential_nested_v16/chungcheong_identity_audit.csv`
- `outputs/active_presidential_nested_v16/regional_identity_audit.csv`
- `outputs/active_presidential_nested_v16/input_manifest.csv`
- `archives/experiments/manual_seed_lineage_v17_rejected_20260728/`
- `outputs/regional_identity_v16_camp_donor_experiment/decision.json`
- `docs/REGIONAL_IDENTITY_V16_20260728.md`
- `archives/experiments/regional_identity_v16_20260728/archive_manifest.csv`
- `outputs/chungcheong_identity_v15_experiment/decision.json`
- `outputs/chungcheong_error_audit_v15/summary.json`
- `archives/experiments/chungcheong_identity_v15_20260728/archive_manifest.csv`
- `outputs/vif_gated_regional_offset_v14_experiment/decision.json`
- `outputs/all_fold_regional_offset_v14_experiment/decision.json`
- `archives/experiments/chungcheong_regional_offset_v14_20260728/`
- `outputs/strategic_lane_transfer_v12_experiment/decision.json`
- `outputs/orientation_affinity_fix_v13_experiment/decision.json`
- `outputs/major_party_core_v11_experiment/decision.json`
- `outputs/regional_accent_regime_v10_ablation/summary.json`
- `docs/CURRENT_MODEL_PERFORMANCE_20260728.md`
- `docs/SPEECH_DERIVED_2017_FAILURE_AUDIT_20260728.md`
- `docs/AUTOMATIC_CANDIDATE_CONTEXT_V2_20260728.md`
- `outputs/speech_derived_candidate_context_v2/decision.json`
- `outputs/speech_candidate_v2_ablation/decision.json`
- `outputs/manual_weight_lineage_audit_v2/input_lineage.csv`
- `docs/AUTOMATIC_THIRD_PRESSURE_V3_20260729.md`
- `outputs/automatic_third_pressure_v3_ablation/decision.json`
- `docs/AUTOMATIC_CANDIDATE_REGIONAL_BASE_V4_20260729.md`
- `outputs/automatic_candidate_regional_base_v4_ablation/decision.json`
- `docs/AUTOMATIC_MEGA_ISSUE_INTENSITY_V5_20260729.md`
- `outputs/automatic_mega_issue_intensity_v5_ablation/decision.json`
- `outputs/automatic_mega_issue_intensity_v5b_event_gate/decision.json`
- `docs/CHUNGCHEONG_ERROR_DIAGNOSIS_20260728.md`
- `outputs/chungcheong_error_audit_v13/summary.json`
- `outputs/predictor_orthogonalization_v14_experiment/decision.json`
- `outputs/direct_party_center_v14_experiment/decision.json`
- `docs/REGIONAL_ACCENT_AND_REGIME_DIAGNOSIS_20260728.md`
- `docs/MAJOR_PARTY_CORE_V11_20260728.md`
- `docs/STRATEGIC_LANE_TRANSFER_V12_20260728.md`
- `docs/SAME_LANE_AFFINITY_V13_20260728.md`
- `archives/experiments/regional_accent_regime_v10_20260728/`
- `archives/experiments/major_party_core_v11_20260728/`
- `archives/experiments/strategic_lane_transfer_v12_20260728/`
- `archives/experiments/same_lane_affinity_v13_20260728/`
- `docs/PARTY_CONTEXT_COHESION_V9_20260727.md`
- `docs/UNIVERSAL_EVIDENCE_PIPELINE_V8_20260727.md`
- `docs/INFORMATION_LEAKAGE_AND_2007_AUDIT_20260727.md`
- `docs/STRUCTURAL_ERROR_AUDIT_V8_20260727.md`
- `archives/experiments/pre_universal_pipeline_v8_20260727/`
- `archives/experiments/universal_pipeline_v8_20260727/`
- `archives/experiments/pre_full_nested_v7_20260727/`
- `archives/experiments/full_nested_v7_20260727/`
- `outputs/active_presidential_nested/government_burden_scores.csv`
- `outputs/active_presidential_nested/contest_regimes.csv`
- `outputs/cumulative_regime_rejection_experiment/decision.json`
- `outputs/incumbent_shock_response/decision.json`
- `outputs/incumbent_shock_response/input_active_v3_predictions.csv`
- `docs/INCUMBENT_SHOCK_RESPONSE_20260719.md`
- `docs/CONTEST_REGIME_GATE_20260719.md`
- `docs/CUMULATIVE_REGIME_REJECTION_20260719.md`
- `docs/STRUCTURAL_LAYER_REACTIVATION_20260719.md`
# 2026-07-29 official-data automation tranche

- Added a credential-safe, paginated, retrying, atomic-cache public-data client:
  `src/news_collector/sources/public_data_api.py`.
- Collected NEC candidate histories through 2022 only. Persisted outputs are in
  `data/raw/official_sources/`; 15/15 candidate-election identities resolved,
  77 history rows, and 13 experimental candidate-region rows.
- The collector reads identity columns only, masks target-election outcomes,
  discards post-2022 rows before checkpoint writes, and never persists the API
  key.
- Added a deterministic official-history regional compiler and strict nested
  ablation. It was **not promoted**: active v16 regional/national MAE remains
  3.3817/1.8417%p versus 3.5865/2.0693%p for official history only.
- Added the official presidential registry collector. The current key is not
  approved for `PofelcddInfoInqireService` and receives HTTP 403; the collector
  is ready to resume after service approval.
- Active model and `data/raw/candidate_regional_base.csv` remain unchanged.
- Full design, source map, and next work:
  `docs/OFFICIAL_DATA_AUTOMATION_ROADMAP_20260729.md`.

## 2026-07-29 district-first correction

- Corrected the candidate-base interpretation: personal and party organization
  evidence is now reconstructed from Assembly constituency results before
  province roll-up.
- Added a cached NEC collector for 1992-2020 Assembly constituency results:
  `scripts/collect_official_assembly_district_history.py`.
- Added `data/raw/official_sources/nec_assembly_district_history.csv` with 1,911
  election-constituency units and 8,484 real candidate rows.
- Fixed an existing parser bug where NEC zero-filled empty candidate slots were
  emitted as candidates.
- Added district-first personal-excess, executive-office, and non-major party
  organization components in
  `presidential_issue_engine/district_reconstructed_candidate_base.py`.
- Strict nested result: regional/national MAE 3.5014/2.0565%p, versus active
  3.3817/1.8417%p. 2007 improves, but 2012 and 2022 regress because one
  constituency can still spread too broadly across a province.
- The district-first layer is retained as an experimental factual compiler but
  is not promoted. Active v16 is unchanged.
- Detailed record:
  `docs/DISTRICT_RECONSTRUCTED_CANDIDATE_BASE_V6_20260729.md`.

## 2026-07-29 clean district-base v8 correction

- The original v6 evaluator was reproducible but not isolated: it also enabled
  role-aware slot assignment and speech-derived candidate context v4.
- Added a clean strict nested evaluator that keeps all active v16 inputs and
  default slot assignment fixed and replaces only the candidate regional-base
  file: `scripts/evaluate_district_candidate_base_clean_v8.py`.
- Input-manifest audit found exactly one changed source hash
  (`candidate_regional_base.csv`) and zero unexpected differences.
- District base alone gives regional/national MAE 3.4429/1.9308%p.
- District base plus balanced 0.60 contest response and the existing
  coefficient-free rejection router gives 3.2501/1.5671%p, winner accuracy
  0.80. 2017 improves from 4.5431/3.2492 to 3.9553/2.5185%p.
- This remains a diagnostic candidate because the 0.60 response was selected
  on the same through-2022 sample and constituency footprint control is still
  pending. Active v16 is unchanged.
- Detailed record: `docs/DISTRICT_CANDIDATE_BASE_CLEAN_V8_20260729.md`.

## 2026-07-31 automatic footprint and response v10

- Diagnosed the 2012/2022 regression as province-wide overexpansion and linear
  double counting of constituency and municipal-office history.
- Added footprint-controlled official candidate base v9: constituency valid
  vote footprint, municipal constituency footprint, winning-office gate, and
  bounded-union repetition. Lost executive candidacies no longer create an
  office base.
- With active response settings, 2012 regional MAE is 2.1987%p and 2022 is
  1.4434%p; the earlier regional regressions are removed.
- Added a strict prior-only automatic contest-response selector. Selected gains
  are 0.50, 0.50, 0.60, 0.633, and 0.65 for 2002-2022 respectively, with every
  target excluded from its selection history.
- Automatic candidate performance is 3.3128%p regional and 1.6185%p national,
  winner accuracy 0.80. Active v16 remains 3.3817/1.8417%p.
- Not promoted: 2007 regional MAE regresses from 4.9237 to 5.1585%p because
  strictly prior official election history does not encode Lee Hoi-chang's
  Chungcheong biographical affinity. Post-election 2008 evidence is forbidden.
- Added `outputs/automation_status_v3/` to separate automatic inputs, remaining
  manual controls, safety bounds, and behavioral parameters.
- Detailed record: `docs/AUTOMATIC_FOOTPRINT_AND_RESPONSE_V10_20260731.md`.

## 2026-08-01 active v17 automation

- Promoted the v10 successor after strict nested ablation. The canonical runner
  is `scripts/run_current_presidential_model.py`; the pointer is
  `data/config/current_presidential_model.json`.
- Active v17 regional/national MAE is 3.2133/1.4815%p, winner accuracy 0.80.
- Candidate regional base is now the footprint-controlled official-history v9
  input. Contest-response gains are selected from earlier outer folds only.
- Chungcheong regional-party mass now uses prior presidential, Assembly PR and
  district, local-council PR and district, and executive-election records.
- Automatic candidate routing adds Lee Hoi-chang in 2007 and Ahn Cheol-soo in
  2017. The dated 2002 policy and 2012 merger facts remain as factual rows.
- Fully automatic recipient routing regressed to 3.4161/1.7910%p, proving that
  election returns alone do not identify every recipient event.
- Automatic third-profile/pressure variants were not promoted because 2017
  regional MAE regressed by 0.373-0.693%p.
- v16 remains unchanged and the rollback checkpoint is
  `backups/model_checkpoints/20260731_v10_automation_start/`.
- Detailed record: `docs/ACTIVE_V17_AUTOMATION_20260801.md`.

## 2026-08-01 active v18 automatic third viability

- Promoted the V14b viability-only ablation as the next V10-lineage model.
  `scripts/run_current_presidential_model.py` now invokes V18.
- V18 replaces third-candidate viability for 2002, 2007, and 2017 with a
  strictly prior election-derived value. Direct-party ballots, district
  organization, prior presidential stature, won offices, and Assembly role
  evidence are kept as separate inputs.
- Manual centrist, anti-major-party, regional-overlap, and third-pressure
  fields remain active. V18 does not claim that these controls are automated.
- V18 regional/national MAE is 3.216549/1.479382%p and winner accuracy is 0.80.
  Relative to V17, regional MAE changes by +0.003280%p and national MAE by
  -0.002073%p. This passes the predeclared automation-equivalence gate but is
  not a performance-improvement claim.
- The V18 nested predictions are byte-identical to the prior V14b ablation.
- Full-profile, automatic-pressure, and simple low candidate-ballot-weight
  experiments were rejected. They demonstrate that direct party preference,
  organization persistence, and candidate personal effect require separate
  channels.
- Immediate rollback checkpoint:
  `backups/model_checkpoints/20260801_pre_v18/`.
- Detailed record:
  `docs/ACTIVE_V18_AUTOMATIC_THIRD_VIABILITY_20260801.md`.

## 2026-08-01 active v20 automatic third character subset

- Ran one-field strict nested ablations for centrist appeal, anti-major-party
  appeal, and regional-base overlap on top of V18 automatic viability.
- Anti-major-party appeal and regional-base overlap passed the predeclared
  automation-equivalence gate separately and together. Centrist appeal failed
  because 2007 regional MAE regressed by 0.089345%p.
- Promoted the passing pair as V20. V20 regional/national MAE is
  3.217252/1.481393%p and winner accuracy is 0.80. This is an automation
  equivalence result, not a performance improvement.
- V20 regenerates all automatic profile stages on every canonical run. Its
  nested prediction hash matches the confirmation ablation.
- V18 rollback checkpoint:
  `backups/model_checkpoints/20260801_active_v18/`.
- Detailed record:
  `docs/ACTIVE_V20_AUTOMATIC_THIRD_CHARACTER_20260801.md`.
- Fixed direct execution of `scripts/run_current_presidential_model.py` by
  registering the repository root before importing the versioned runner. The
  canonical command now exits successfully and reproduces the V20 prediction
  hash.

## 2026-08-01 regional party channel experiments

- Restored the Liberal Democrats/Jaminryun lineage from NEC constituency rows
  and split direct party preference, district organization, and candidate
  proxy evidence.
- Direct replacement V19 regressed to 3.221768/1.546870%p and was rejected.
- Reliability-only corroboration V19b produced 3.219354/1.470656%p. It improved
  the national diagnostic but regressed regional macro MAE, so it was not
  promoted.
- An initial apparent V19b pass was invalid because the evaluator omitted the
  explicit V18 automatic profile path. The corrected evaluator and canonical
  candidate now have identical prediction hashes. Only corrected values are
  authoritative.
- The factual party-lineage compiler remains shadow infrastructure. Active V20
  retains the V18 full-history regional reservoir.
- Detailed record: `docs/REGIONAL_PARTY_CHANNELS_V19_20260801.md`.

## 2026-08-01 remaining V20 automation boundary

- Corrected the automation inventory: V20 viability, anti-major appeal, and
  regional overlap are automatic for 2002/2007/2017 but the 2012 final-minor
  and 2022 withdrawn-candidate rows remain curated.
- The 2022 gap is caused by the target registry omitting a separate preliminary
  `pres_2022` Ahn row even though the official history cache contains the person
  and withdrawal records.
- Added concrete high-, medium-, and lower-feasibility automation plans in
  `docs/AUTOMATION_REMAINING_V20_20260801.md`.

## 2026-08-02 historical V21 unified exact genealogy

- V21 was the active model at this stage. V20 is preserved at
  `backups/model_checkpoints/20260802_pre_active_v21`.
- One exact-party ledger now supplies both the regional-identity layer and the
  final analytic-bloc projection used by Ridge. There is no separate generic
  regional-history estimator.
- Exact NEC Assembly constituency party rows replace collapsed constituency
  rows. All regions and election types use one estimator; election type changes
  evidence reliability only.
- Dated party rename/merger facts are traversed with one predecessor-successor
  graph. No transition weight is fitted to presidential outcomes.
- Manual candidate-alignment rows are not read. The 2002 policy commitment is
  not misclassified as party ancestry.
- Strict nested regional/national diagnostic MAE is
  `3.395154%p`/`1.765729%p`; winner accuracy is `80%`.
- This is a methodology-first promotion. V20 remains retrospectively more
  accurate by `0.177902%p` regional and `0.284336%p` national MAE.
- Validation: `457 passed`, strict PIT PASS, 10/10 fold guards PASS, no 2025
  path in the active input manifest, and canonical/ablation prediction hashes
  match.
- Detailed record:
  `docs/ACTIVE_V21_UNIFIED_EXACT_GENEALOGY_20260802.md`.

## 2026-08-02 historical V22 automatic controls

- V22 was the active model at this stage. The pre-change checkpoint is
  `backups/model_checkpoints/20260802_pre_automatic_policy_v22`.
- The promoted minimum bundle is automatic policy-commitment strength,
  Assembly-derived mega-issue taxonomy/intensity, automatic third-candidate
  profile/withdrawn landscape/source-lane pressure, and automatic
  economic/housing responsibility input tables.
- Third-candidate pressure is not valid as an isolated replacement. With the
  retained V20b profile it worsened regional MAE to `3.429772%p`; the coherent
  automatic profile+landscape+pressure bundle is required.
- Strict nested regional/national diagnostic MAE is
  `3.322926%p`/`1.677567%p`; winner accuracy remains `80%`.
- Versus V21, regional and national MAE improve by `0.072228%p` and
  `0.088161%p`. V20 remains better by `0.105674%p` regional MAE.
- Election changes versus V21: 2002 `-0.097982%p`, 2007 `-0.364129%p`,
  2012 unchanged, 2017 `+0.100391%p`, 2022 `+0.000578%p`.
- The 2002 policy registry stores a dated factual administrative-capital
  commitment but no effect size. The compiler derives strength from candidate
  issue association, election-normalized issue importance, and source confidence.
- Generation composition remains shadow because official lagged reports are
  sparse and slightly worsened MAE. Behavioral party-retention remains shadow
  because the four transition rows have no usable before/after direct-party pair.
- Withdrawal transfer rate and voter compliance remain semiautomatic scenarios.
- Selection among V22 bundles used 2002-2022 development outcomes; V22 is not an
  untouched holdout claim.
- Validation: `462 passed`, strict deep PIT PASS, outcome invariance `215/215`,
  no 2025 path or row, and canonical/ablation hashes match.
- Detailed record: `docs/ACTIVE_V22_AUTOMATIC_CONTROLS_20260802.md`.

## 2026-08-02 active V23 unified withdrawal and generation controls

- V23 is the active model. V22 remains intact, and the pre-change checkpoint is
  `backups/model_checkpoints/20260802_pre_v23_unified_withdrawal_generation`.
- One canonical candidate profile now supplies final-third and withdrawn-
  candidate traits. The 2022 Ahn profile is derived from strictly prior 2017
  same-person evidence and shrunk toward neutral by evidence strength.
- The 2012 Ahn fallback now uses pre-withdrawal Assembly target-mention breadth
  for stature only. Centrist and anti-major traits remain neutral because the
  attention evidence contains no valid direction label.
- `withdrawal_transfer_registry.csv` is the only active behavioral transfer
  registry. Event facts remain in `withdrawal_events.csv`; the three legacy
  transfer CSVs are isolated and absent from the active input manifest.
- Generation composition uses the latest strictly prior official report. The
  2022 fold therefore uses the 2017 report, never the 2022 post-election report.
- Strict nested regional/national diagnostic MAE is
  `3.367899%p`/`1.597845%p`; winner accuracy remains `80%`.
- Versus V22, regional MAE changes by `+0.044973%p`, national MAE improves by
  `0.079723%p`, and the largest election regression is 2012 at `+0.207022%p`.
- Validation: `471 passed`, V22 exact reproduction, conservative promotion gate PASS, V23
  active/experiment prediction SHA match, legacy transfer inputs absent,
  strict deep PIT PASS, and V23 outcome invariance `215/215`.
- The low/medium/high withdrawal transfer scenarios remain semiautomatic. They
  are universal rules rather than election-specific freely tuned point values.
- Selection still used 2002-2022 development outcomes. This is not an untouched
  historical holdout or a claim based on 2025.
- Detailed record:
  `docs/ACTIVE_V23_UNIFIED_WITHDRAWAL_GENERATION_20260802.md`.

## 2026-08-02 V24 regional-shape and interval shadow experiment

- V23 remains frozen and active. V24 is not promoted.
- The V24 evaluator correctly uses `layer_pred`, the V23 active point-prediction
  column. `pred` is an intermediate column and must not be used to reproduce
  the frozen V23 metrics.
- A national-total-preserving regional accent tilt was evaluated with only
  strictly prior region-volume weights. The target election's realized turnout
  is used only for post-hoc error diagnostics.
- With the conservative nested gain grid `0, 0.025, 0.05`, regional macro MAE
  changes from `3.367899%p` to `3.344180%p`; national macro MAE changes from
  `1.597845%p` to `1.597168%p`; winner accuracy stays `80%`.
- The `0.023719%p` regional gain is too small for promotion, and 2012 and 2022
  regress slightly. Wide gains through 0.3 showed clear early-fold overreaction.
- Hierarchical residual intervals were evaluated with candidate-common,
  regional, and local components. Regional row intervals remain wide and are
  rejected for production.
- National candidate intervals are now aggregated separately with strictly
  prior region weights. A coherent empirical hierarchy at scale `0.75`, with
  region vote-volume uncertainty, gives equal-election coverage/mean widths of
  `91.67%/8.42%p`, `100%/9.79%p`, and `100%/12.48%p` for the 90%, 95%, and 99%
  levels. This is a shadow candidate, not a promoted calibration.
- Region-weight uncertainty is estimated only from earlier vote-volume
  transitions and automatically shrinks to zero when none exist. Its addition
  changes the 95% national width only from `9.76%p` to `9.79%p`.
- The interval output is a predictive interval, not a classical confidence
  interval. No separate Ridge covariance draw is yet added; nested residuals
  implicitly include historical estimation error.
- V23 input SHA remains
  `dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b`.
- Validation after the structural-uncertainty extension: `476 passed`.
- Detailed record:
  `docs/EXPERIMENT_V24_REGIONAL_SHAPE_INTERVALS_20260802.md`.

## 2026-08-03 V24 national 95% interval refinement

- The coherent single-distribution national reference remains empirical full
  hierarchy at scale `0.75`. With 10,000 draws its 95% equal-election coverage
  is `100%` and mean total width is `9.79%p`.
- A five-seed, 10,000-draw refinement selected level-specific scales
  `90%=0.74`, `95%=0.72`, and `99%=0.60`.
- The selected 95% shadow interval has mean total width `9.42%p`, approximately
  `+/-4.71%p`, and minimum seed coverage `100%` on four evaluable elections.
- Scale `0.71` was rejected because its coverage changed with Monte Carlo seed;
  scale `0.70` also undercovered.
- Level-specific intervals are forced to nest: 95% contains 90%, and 99%
  contains 95%. The final nested 99% width is `10.12%p`.
- This is development-outcome-aware and remains shadow. It is not a classical
  coefficient confidence interval or an untouched-holdout calibration.
- Detailed record:
  `docs/EXPERIMENT_V24_INTERVAL_SCALE_REFINEMENT_20260803.md`.

## Development-order caveat for the 2025 target (2026-08-21)

Two corrections applied to the 2025 path on this date are defensible on their
own evidence, but the order in which they were found is not a pre-registered
forecast and the record should say so plainly.

What was corrected:

- `mega_issue_terms.csv` carried nine dedicated pres_2017 crisis terms and none
  for pres_2025, so the institutional-crisis severity was measured with an empty
  instrument. Eight terms were registered, all scoped to pres_2025 alone, all
  drawn from events between 2024-12 and 2025-04 and therefore inside the D-1
  cutoff. Historical mega-issue controls are byte-identical afterwards.
- `weak_same_lane_refusal` applied a flat 0.50 desertion rate to every weak
  third candidate although a continuous carry-in measure already existed. The
  rate now falls linearly to zero at the exemption threshold already used by
  `third_candidate_lineage_constraint`. Both anchors predate the change.

Why each is defensible without reading the 2025 result: a term table with nine
entries for one election and zero for another is a coverage defect on its face,
and a flat rate alongside a continuous measure is an internal inconsistency on
its face. No parameter was fitted to the realised 2025 composition, and the
scored panel is bit-identical, including the refusal rate for 권영길 2002 and
심상정 2022.

What is not clean: the investigation began because the 2025 composition looked
wrong, so the search direction was outcome-informed even though the fixes were
not. A proposal to scale the crisis intensity down in proportion to the measured
severity was raised on that basis and withdrawn once the measurement gap was
identified, but the selection of which defects to pursue, of the 0.02 threshold,
and of the eight registered terms all occurred after the result was known.

This does not make the 2025 output a scored prediction and it must still not be
compared with the realised result for selection or tuning. It does mean the 2025
path carries a weaker methodological guarantee than the frozen historical panel,
and any external claim should describe it as a corrected demonstration rather
than an out-of-sample forecast.

## The event-class alignment is forecast-only and unmeasurable on the panel (2026-08-21)

Two candidate changes to the direct mega-issue attribution were tested and both
were rejected on evidence. The measurements are recorded here because the second
one settles a question that has been raised more than once.

### Rejected: gating `direct_mega_score` by `target_specificity`

`direct_mega_score` multiplies direction, association strength, confidence and
intensity, and uses none of the five measured quality axes of the taxonomy. It
is natural to ask whether a low-quality measurement should attribute less, and
`target_specificity` is the axis that most directly expresses that. Adding it as
a factor changes exactly one scored election, because it is the only one that
reaches the attribution at all:

| | regional macro | level macro | 2017 burdened candidate |
| --- | --- | --- | --- |
| current | 3.440 | **0.990** | **+0.030 %p** |
| specificity-gated | 3.433 | 1.013 | +0.441 %p |

The regional gain of 0.007 is noise from redistribution; the level metric gets
worse and the single historical calibration point moves from essentially exact
to fourteen times that error. Not adopted.

### Verified: the discontinuous alignment is the correct shape

`align_profile_to_event_class` keeps only issues declared compatible with the
election's shock class. Three separate facts about it were established.

**It is reached from the prospective forecast only.** `run_prospective_forecast`
patches it in around `compile_direct_mega_scores`; the retrospective calls that
function on the raw profile. The scored panel therefore never runs the alignment.

**It is provably inert on the scored panel anyway.** Applying it to the
retrospective reproduces every reported number bit-identically. Two independent
reasons: the elections whose winning issue is off-class (2007 `security_nk`,
2022 `security_nk`) sit at intensity 1.00, where `intensity_activation` is
exactly zero, and the one election above the gate already selects an on-class
issue. So the calibration measured on the unaligned path transfers unchanged,
but the panel also cannot measure this component at all. It is a declared
assumption. `tests/test_mega_issue_adjustment.py` pins both halves.

**Making it continuous moves the discontinuity rather than removing it.** With a
graded weight `lam` on off-class issues, the winner-take-all issue race flips at
the point where the two selection scores cross:

| election | on-class best | off-class best | flips at |
| --- | --- | --- | --- |
| 2002, 2012, 2017 | — | — | never (`lam` > 1) |
| 2007, 2022 | 0 | 0.2318 / 0.3962 | already zero by intensity |
| **2025** | 0.5137 | **0.8534** (`withdrawal_event`) | **`lam` = 0.602** |

Every value of `lam` in [0, 1] gives an identical retrospective, so a sweep on
the panel reports "no change, safe" while 2025 jumps at 0.602 from
`regime_change` (score −0.3213) to `withdrawal_event` (−0.4414, rising to the
−0.50 score cap at `lam` = 1). Downstream, `apply_direct_mega_shift` merges with
`validate="many_to_one"`, so a second surviving issue raises rather than blends:
the selection is a structural requirement, not a stylistic choice. A graded
weight therefore keeps a step function and only relocates the step from a
semantic boundary to an arbitrary crossing point, beyond which the whole shock
transfers to an unrelated issue.

Direction matters for interpreting the 2025 composition: the alignment is what
holds slot B up, not what pushes it down. Removing it costs that candidate about
1.6 %p. Concerns that the 2025 forecast suppresses the conservative slot too
hard are not addressed by loosening this layer, which makes the suppression
stronger.
