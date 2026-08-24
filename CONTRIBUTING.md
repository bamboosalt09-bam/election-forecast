# Contributing and extending the forecast engine

Contributions are welcome, but frozen model versions are immutable evidence.
Do not edit files under `outputs/active_presidential_nested_v23` through `v28`,
their versioned fixed datasets, or their finalization manifests in place.

## Development setup

```bash
python -m pip install -e ".[dev,viz]"
python -m pytest -q
python scripts/audit_public_active_presidential_model_v30.py
```

Before submitting a change, run `python scripts/audit_github_baseline.py` as
well. A change that intentionally alters model output must be implemented as a
new version with a new runner, output directory, promotion record and audit.

## Supported extension points

### Add or inspect inputs

Use `python scripts/describe_inputs.py inventory`, `check ELECTION_ID`, and
`sources ELECTION_ID` before changing a loader. New time-varying rows require
an `available_date`; source URLs, derivation notes and confidence fields should
be retained where the schema supports them. Templates live under
`presidential_issue_engine/fixed_dataset/templates`.

### Add a new election

Keep the target outcome outside feature construction. Define the ballot and
candidate registry as of the forecast cutoff, reconstruct PIT-safe context,
and assert region/candidate coverage. A prospective target must not be added to
the scored development panel merely to make the runner accept it.

### Change feature assembly or Ridge inputs

The canonical assembly and regression paths are in
`presidential_issue_engine/issue_vote_engine.py`. Add predictor-leakage,
outcome-mutation and chronological tests for any new field. Slot labels or
target results must never become predictors.

### Add a structural or postprocess layer

Implement the transform as a separate module with an audit table. Document its
input column, output column, conservation behavior and position in the stack.
Test it in isolation and in order permutations where recipient weights depend
on prior stages. Promote it only through a new version wrapper.

### Replace issue or stance classification

Preserve sentence ownership, target attribution and direction as separate
stages. Keep frozen adjudication sets unchanged, report risk versus coverage,
and retain a deterministic fallback for unavailable optional model weights.

### Add a visualization

Read finalized artifacts rather than reimplementing model calculations. Record
external geometry or artwork with a pinned source, hash, license and required
attribution in `docs/VISUALIZATION_DATA.md` and `NOTICE`.

## Pull-request evidence

A model-affecting pull request should include the hypothesis, isolated
ablation, per-election effects, conservation checks, frozen-boundary audit and
an explicit statement of whether any evaluated outcome informed development.
Historical improvement alone is not sufficient evidence for future accuracy.
