## Change

- What changed:
- Why it changed:

## Validation

- [ ] `python scripts/audit_github_baseline.py`
- [ ] `python -m pytest -q`
- [ ] Active-model audit, when the forecast path is affected
- [ ] Strict chronological nested evaluation, when model behavior is affected

## Information boundary

- [ ] No 2025 presidential outcome was used for fitting, tuning, ablation, or comparison
- [ ] Every new input has an `available_date` or an equivalent point-in-time rule
- [ ] No secret, local cache, backup, or generated bulk output is included

## Model lifecycle

- [ ] Frozen V23-V27 artifacts remain unchanged, or a new version and promotion record are included
- [ ] `current_presidential_model.json` and its public compatibility alias still select the same version
- [ ] Performance changes are reported using the same metric definitions as the active baseline
