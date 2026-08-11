# Precision-first Assembly stance classifier

## Status

This classifier is a shadow artifact. It is not connected to active forecast
inputs or score computation.

- model: `stance_precision_first_v1`
- data boundary: presidential-election text associated with 2002-2022 only
- vote outcomes used: no
- active forecast changed: no
- adoption gate: failed

## Error objective

The classifier treats mistakes asymmetrically.

1. Neutral to negative/positive is harmful.
2. Negative to positive or positive to negative is harmful.
3. Negative/positive to neutral is an abstention and is less harmful.

Directionality and polarity are fitted as separate logistic heads. A row is
emitted as negative or positive only when both heads clear their thresholds.
Questions, quotations, mixed cues, negation conflicts, and metalinguistic
examples receive an additional direction threshold. Everything else abstains
to neutral.

Neutral does not mean discarded. Each abstained row retains a separate
information score and category based on numeric evidence, reporting language,
procedural content, analytical contrast, and length. These fields do not add a
positive or negative direction.

## Inputs

| Input | Rows | Use |
|---|---:|---|
| locked gold | 243 | model selection and engineering evaluation |
| development split | 186 | grouped 5-fold OOF selection |
| engineering holdout | 57 | fixed-policy audit after selection |
| approved weak anchors | 633 | low-weight training support only |
| manually reviewed expansion | 122 | training augmentation only |

The weak anchors contain only explicit direct criticism, explicit direct
support, anti-corruption policy descriptions, anti-corruption praise, and
reported defense. They receive `weak_confidence * weak_scale`; the selected
weak scale is `0.02`. Gold and weak text hashes are required to be disjoint.

The selected representation uses the current sentence only, with candidate or
party names masked. Character and word TF-IDF features are combined with
UTF-8 Korean structural features. Selected parameters:

- logistic `C`: `0.50`
- direction threshold: `0.525`
- polarity threshold: `0.725`
- risk surcharge: `0.20`

## Results

| Metric | Development OOF | Engineering holdout |
|---|---:|---:|
| neutral to direction | 0 / 75 | 0 / 23 |
| wrong direction | 0 / 111 | 0 / 34 |
| correct directional emissions | 18 | 4 |
| directional coverage | 16.22% | 11.76% |
| direction to neutral | 83.78% | 88.24% |
| observed directional precision | 100% | 100% |
| harmful-error one-sided 95% upper bound | 15.33% | 52.71% |

The observed harmful error count is zero, but this is not enough evidence to
claim a safe classifier. Only 18 and 4 directional predictions were emitted.
The original engineering gate required all of the following:

- zero observed harmful errors;
- harmful-error one-sided 95% upper bound at most 1%;
- correct directional coverage at least 40%.

The 40% threshold had no statistical or operational derivation. It was removed
on review. For a selective classifier, low coverage means more abstentions and
less utility; it does not itself make emitted labels unsafe. The current gate
therefore treats coverage as a diagnostic only and requires:

- a newly locked independent audit, not the reused engineering holdout;
- zero observed neutral-to-direction or sign-reversal errors;
- a one-sided 95% harmful-error upper bound at most 5%, which requires at
  least 59 independently audited emissions when zero errors are observed;
- audited target attribution, strict point-in-time compliance, and no strict
  rolling degradation after integration;
- an absolute per-row vote-share effect cap of 0.1%p.

The 5% label-risk bound is an explicit operational tolerance, not an estimated
natural constant. Coupled with the 0.1%p effect cap, its upper-bound expected
signed impact is 0.005%p per affected row. The policy is implemented in
`stance_adoption_assessment`; no current artifact passes because a new locked
audit and integration non-degradation test have not yet been completed.

The current classifier fails the latter two requirements and remains shadow.
With zero observed errors, at least 299 independently audited directional
emissions are needed before the 95% upper bound can fall below 1%.

## Second improvement attempt

The following alternatives were evaluated after the first precision model:

| Candidate | Development coverage | Holdout coverage | Holdout harmful errors | Decision |
|---|---:|---:|---:|---|
| TF-IDF precision v1 | 16.22% | 11.76% | 0 | child model |
| LinearSVC selective model | 16.22% | 11.76% | 0 | no gain |
| all 5,000 weak rows at reduced weights | 16.22% | not selected | - | no gain |
| fixed Korean sentence embedding | 8.11% | 5.88% | 0 | worse alone |
| relaxed two-model consensus | 21.62% | 20.59% | 2 | rejected |
| hard-risk consensus ensemble v2 | **19.82%** | **14.71%** | **0** | best shadow |

The embedding child uses the fixed `jhgan/ko-sroberta-multitask` encoder at
revision `8fca7c9c98c26599be0e14b9916b11a756a26f19`. The 768-dimensional encoder
is not fine-tuned. Only logistic heads and abstention thresholds are learned.
Its dependencies are isolated under `.venv-stance`; the active project Python
environment is unchanged.

The accepted shadow ensemble first takes the conflict-safe union of the two
children. It then adds a prediction only when both children agree on the sign,
their joint direction and polarity scores clear the development-selected
thresholds, and the sentence has no question, quotation, mixed-negation, or
metalinguistic risk flag. Relaxed consensus without hard risk exclusion caused
two neutral-to-positive errors in the historical engineering holdout and was
rejected.

Ensemble v2 results:

| Metric | Development OOF | Engineering holdout |
|---|---:|---:|
| neutral to direction | 0 / 75 | 0 / 23 |
| wrong direction | 0 / 111 | 0 / 34 |
| directional coverage | 19.82% | 14.71% |
| direction to neutral | 80.18% | 85.29% |
| harmful-error one-sided 95% upper bound | 12.73% | 45.07% |

This is a real but insufficient improvement. Ensemble v2 remains inactive
because the historical holdout is not an independent adoption audit and the
integration checks have not been completed.

## Third improvement attempt: direct-target gold expansion

The 5,000-row source pool was audited for real parliamentary sentences that
directly concern a candidate or party. A conservative manual pass added 122
rows rather than converting weak labels into gold:

| Dimension | Rows |
|---|---:|
| person target | 87 |
| party target | 35 |
| negative | 57 |
| neutral | 43 |
| positive | 22 |

Each row records the target-correctness decision and review basis in
`data/shadow/stance_manual_gold_expansion_v1.csv`. Text hashes are unique and
must not overlap the original gold. Only elections from 2002 through 2022 are
allowed. During grouped OOF evaluation, expansion rows from the validation
row's `source_file`, `meeting_date`, and `committee` group are excluded from
that fold. The resulting manual training counts were 121, 121, 120, 122, and
122 rows across the five folds.

The augmented TF-IDF child and the fixed embedding child were then combined
with the same hard-risk abstention rule. Results are:

| Candidate | Development coverage | Holdout coverage | Observed harmful errors | OOF upper 95% | Holdout upper 95% |
|---|---:|---:|---:|---:|---:|
| TF-IDF precision v1 | 16.22% | 11.76% | 0 | 15.33% | 52.71% |
| hard-risk ensemble v2 | 19.82% | 14.71% | 0 | 12.73% | 45.07% |
| augmented TF-IDF v3 | 19.82% | 17.65% | 0 | 12.73% | 39.30% |
| augmented ensemble v4 | **23.42%** | **20.59%** | **0** | **10.88%** | **34.82%** |

The v4 gain over the first model is 7.20 percentage points in grouped OOF
coverage and 8.82 percentage points on the engineering holdout. This is an
improvement in selective coverage, not proof of a low error rate. The holdout
has been reused during prior engineering comparisons and is not an independent
confirmatory test. A three-child union was also inspected and is not selected:
it was discovered after viewing holdout behavior and therefore remains an
exploratory result.

Augmented ensemble v4 remains shadow under the reviewed gate. It has not passed
a newly locked audit, target-attribution audit, or rolling non-degradation
test, and does not change forecast inputs, issue weights, or vote-share output.

## Neutral information retention

- development abstentions: 168; nonzero information score: 165
- engineering holdout abstentions: 53; nonzero information score: 52

The information score is diagnostic only. It must not be used as a signed
candidate or party effect.

## Known limitation

The original gold set was not sufficient for candidate/party attitude learning:

| Target type | Gold rows |
|---|---:|
| none | 181 |
| government | 53 |
| person | 5 |
| party | 4 |

The expansion raises the combined training pool to 92 person-target and 39
party-target rows. This is materially better but still small, especially after
splitting by election, target, quotation ownership, and sign. Further work must
use a new locked audit set; the historical engineering holdout must not be used
for another selection decision.

## Reproduction

```powershell
python scripts\train_stance_precision_first.py
pytest -q tests\test_stance_precision.py

# Optional isolated embedding experiment
.\.venv-stance\Scripts\python.exe scripts\train_stance_embedding_precision.py --local-files-only
python scripts\build_stance_precision_ensemble.py

# Manually reviewed direct-target expansion and final shadow ensemble
python scripts\train_stance_precision_augmented.py
python scripts\build_stance_precision_ensemble.py `
  --first-dir outputs\assembly_stance\precision_augmented_v3 `
  --second-dir outputs\assembly_stance\precision_embedding_v1 `
  --output-dir outputs\assembly_stance\precision_augmented_ensemble_v4 `
  --model-version stance_precision_augmented_ensemble_v4
```

Artifacts are under:

- `outputs/assembly_stance/precision_first_v1`
- `outputs/assembly_stance/precision_embedding_v1`
- `outputs/assembly_stance/precision_ensemble_v2`
- `outputs/assembly_stance/precision_augmented_v3`
- `outputs/assembly_stance/precision_augmented_ensemble_v4`
