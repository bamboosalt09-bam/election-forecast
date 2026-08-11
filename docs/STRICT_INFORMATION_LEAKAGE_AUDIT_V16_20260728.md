# Strict Information-Leakage Audit: Active V16

Date: 2026-07-28

Active policy: `active_strict_nested_v16_regional_identity`

Active output: `outputs/active_presidential_nested_v16`

Follow-up status: an attempted v17 removal was rejected and fully rolled back.
The first ablation removed both manual ancestry and unavailable Assembly-derived
posture during regeneration, so its performance change was confounded. The
active policy remains v16 while the lineage question is reviewed without model
changes. Rejected artifacts are preserved under
`archives/experiments/manual_seed_lineage_v17_rejected_20260728`.

## Executive conclusion

The Ridge outer folds exclude the target election, and the existing deep
target-outcome mutation test confirms that changing a target election's vote
share does not change its regional row predictions. That part is
point-in-time safe.

The full active system is not yet a clean strict forecast backtest. One
transitive-lineage defect is confirmed, while two reporting/provenance
boundaries need to remain explicit:

1. `manual_issue_seed_enabled=false` does not remove manual issue seed
   information from transitive derived inputs.
2. National backtest aggregation legitimately uses realized target-election
   `contest_votes`, but that output is conditional on realized turnout and must
   not be confused with a pre-election turnout forecast.
3. The 2002 two-way scoring scope is substantively consistent with the
   preliminary stature and withdrawal model, but its final exclusion is stored
   as a manually fixed rule rather than generated from that model.

In addition, several candidate, regional, third-candidate, withdrawal, issue,
and regime gains were authored or selected after the 2002-2022 outcomes were
known. This is development-set reuse rather than a direct target-row leak, but
it means the five-election score is not an untouched out-of-sample estimate.

Do not increase regional-identity gains against the same five election
outcomes until the defects below are corrected and the gain is selected only
from prior-fold or non-presidential evidence.

## Finding 1: target-turnout aggregation is a valid evaluation diagnostic

Severity: no predictor leakage; labeling and deployment boundary only.

`scripts/evaluate_preliminary_slot_shadow_nested.py::_metrics` calculates each
candidate's national prediction with:

```python
weights = group["contest_votes"]
pred = np.average(group[prediction_column], weights=weights)
```

`contest_votes` is the realized valid-vote volume in each target-election
region. Using it after prediction is a valid way to compare regional predicted
shares with the realized national result. It gives populous regions the proper
influence and is preferable to an equal-region average. It does not change the
regional predictions, model coefficients, feature values, or fold selection.
Therefore:

- `regional_equal_election_macro_mae_pp=3.381670` is a valid evaluation metric.
- `national_equal_election_macro_mae_pp=1.841654` is also a valid historical
  backtest metric: it measures the national share implied by the regional
  predictions under the turnout distribution that actually occurred.
- The same number is not a standalone pre-election national forecast because
  the eventual regional turnout distribution was not yet known at forecast
  time.

Recommended reporting split:

- Keep the current target-turnout-weighted metric as the primary historical
  comparison requested by the project.
- Optionally produce `national_predictions_ex_ante.csv` using a frozen
  forecast-date regional electorate or prior-turnout weight when evaluating
  deployability before election day.
- Label the current calculation
  `national_predictions_posthoc_target_turnout.csv` to prevent the two
  questions from being conflated.
- Put the aggregation basis in every metric column name and manifest entry.

## Finding 2: the 2002 two-way target is coherent but not fully derived

Severity: provenance and reproducibility gap; not a confirmed information
leak.

The 2002 model intentionally estimates Chung Mong-joon's pre-withdrawal stature
and then removes him after the Roh-Chung unification. The current preliminary
assignment gives Chung:

- pre-withdrawal expected share: `24.5542%`
- `third_viability_input=1.0`
- post-withdrawal status: `transfer_reservoir`
- post-withdrawal expected share: `0.0%`

This supports the intended ex-ante story: Chung was a substantial preliminary
candidate, unified with Roh, and was removed before the final contest. Kwon
Young-ghil remained on the actual ballot but was outside the model's declared
major competitive contest. Scoring Roh and Lee on a two-way normalized basis
is therefore a defensible estimand and is not, by itself, outcome leakage.

The remaining implementation weakness is narrower. The result table uses
source slot `C` for Kwon, while the preliminary layer temporarily uses that
container for Chung. The final Kwon exclusion is stored separately in
`scored_contest_scope.csv`; it is not mechanically emitted by the preliminary
stature and withdrawal classifier. The rule may be based on pre-election
knowledge, but the current artifact does not prove that derivation.

Required correction:

- Keep the current two-way 2002 estimand.
- Generate its scope from the frozen preliminary candidate-status and viability
  rule, or explicitly classify it as a predeclared evaluation estimand.
- Use stable candidate IDs in the preliminary and ballot registries so Chung
  and Kwon can be represented without relying on a shared source-slot
  container.
- Report the two-way competitive-contest metric separately from an optional
  all-ballot metric.

## Finding 3: manual issue seed is still active through descendants

Severity: confirmed configuration and lineage failure.

The active policy says `manual_issue_seed_enabled=false`, which blocks direct
loading of the manual enhanced issue CSVs in the main issue compiler. It does
not block their descendants:

- `scripts/build_candidate_party_speech_context.py` reads
  `data/raw/candidate_issue_profile.csv`.
- `scripts/build_candidate_party_tone_gap.py` reads both
  `candidate_issue_profile.csv` and `mega_issue_attribution.csv` and uses their
  direction, polarity, weight, and confidence.
- `scripts/build_candidate_public_treatment.py` reads the same manual files and
  uses them in legitimacy, negative treatment, fatigue, and protest scores.
- `presidential_issue_engine/build_candidate_vote_conversion_context.py`
  consumes the party-context, tone-gap, and public-treatment outputs.
- `scripts/build_through2022_automatic_issue_seeds.py` consumes party tone and
  public treatment. Candidate treatment does not create its issue direction,
  but it remains in the generated schema and supplies its candidate base.
- The active postprocess always applies party-context and public-treatment
  adjustments and uses `conversion_scale=0.05`.

The active input manifest confirms that all of those derived files are read.
Thus the manual seed is disabled only as a direct input, not as an information
source.

Required correction:

- Build an explicit artifact dependency graph with source classes.
- When manual seed is disabled, reject any artifact whose ancestry includes
  `candidate_issue_profile.csv` or `mega_issue_attribution.csv`.
- Rebuild party context, public treatment, conversion context, and automatic
  issue seeds from assembly-derived inputs only.
- Store the dependency hash and ancestry list in every generated artifact.

## Finding 4: historical manual values are not provenance-verifiable

Severity: strict-backtest failure unless treated as retrospective development
features; not proven target-row arithmetic leakage.

The current manually curated candidate and regional files generally have an
`available_date`, but no `frozen_at` and usually no `source_url`. Examples:

| File | Rows | `source_url` | `frozen_at` |
|---|---:|---:|---:|
| `candidate_issue_profile.csv` | 20 | no | no |
| `mega_issue_attribution.csv` | 10 | no | no |
| `candidate_regional_base.csv` | 11 | no | no |
| `third_candidate_profile.csv` | 5 | no | no |
| `withdrawn_candidate_transfers.csv` | 3 | no | no |

These files were created in 2026 for historical elections. An election-eve
`available_date` can show that the underlying fact could have been known, but
it cannot show that the numeric score was fixed before the outcome was seen.
The values must therefore be labeled retrospective development inputs unless
they are mechanically reconstructed from dated source records under a frozen
formula.

Required correction:

- Add `source_url`, `observed_at`, `frozen_at`, `derivation_version`, and
  `provenance_class`.
- Permit `strict_backtest` only for `prospective_frozen` or deterministic
  source-derived values.
- Keep retrospective values available for model research, but exclude them
  from strict performance claims.

## Finding 5: full-sample architecture and gain selection

Severity: development-set reuse and optimistic model assessment; not a direct
fold-computation leak.

The base Ridge selection trace is rolling: each target fold uses only prior
elections for fitting and configuration selection. However, v8-v16 stage
definitions and several numeric gains were developed while all 2002-2022
outcomes were visible. The active manifest already records:

- `untouched_historical_holdout=false`
- `strict_nested_postprocess_selection=false`
- `direct_mega_gain_development_selected=true`
- `candidate_numeric_parameters_historically_development_selected=true`

The v16 general regional-identity gain was also compared on all five elections.
A larger gain may be a useful research setting, but selecting it by the same
five outcomes cannot improve the credibility of the strict score.

Required correction:

- Select every gain inside the outer fold using only earlier elections, or
  estimate it from prior non-presidential proportional/local elections.
- Record a per-fold `training_elections`, `selection_elections`, and artifact
  cutoff manifest.
- Treat 2002-2022 as a development corpus after these repeated iterations.
- Reserve a genuinely unseen future election, or a pre-registered historical
  reconstruction, for the final performance claim.

## Why the current deep audit still passes

The existing mutation audit is useful but narrower than its name suggests. It
detects direct use of target vote-share values in the regional predictor. It
does not detect:

- target-turnout use after regional prediction,
- a result-derived or collision-prone candidate universe,
- a manual source hidden behind precomputed descendants,
- backdated subjective values without a freeze record,
- architecture and gain choices made after inspecting target errors.

Its `PASS` should be renamed to
`regional_target_vote_value_invariance_pass`, not interpreted as a full
information-leakage clearance.

## Correction order

1. Freeze v16 and preserve its current output as a development snapshot.
2. Split ex-ante national aggregation from target-turnout posthoc diagnosis.
3. Make the 2002 two-way scope reproducibly derive from the pre-election
   stature and withdrawal policy.
4. Enforce transitive manual-seed blocking and rebuild all descendants.
5. Add provenance classes and reject unverifiable manual values in strict mode.
6. Move every adjustable gain into prior-only nested selection or external
   non-presidential calibration.
7. Rerun strict metrics and only then test larger regional-identity gains.

## Current claim boundary

The current v16 numbers describe performance of a retrospective development
system with target-excluded Ridge fits and point-in-time regional predictors.
They do not yet describe a fully prospective, untouched election forecast.
The regional error remains useful for engineering comparison. The national
target-turnout metric is a post-election diagnostic, and the full-system score
must be recomputed after contest-scope provenance and transitive-lineage repair.
