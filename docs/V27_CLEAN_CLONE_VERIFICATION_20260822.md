# V27 clean-clone verification — 2026-08-22

## Scope

A new clone of branch `codex/v27-release-readiness` at commit `5013e83` was
created under the operating-system temporary directory. A new virtual
environment was created inside that temporary boundary; no editable package
from the development checkout was reused.

## Commands represented

```text
git clone --branch codex/v27-release-readiness --single-branch REPOSITORY TEMP/repo
python -m venv TEMP/.venv
TEMP/.venv/python -m pip install -e "TEMP/repo[dev,reproduce-v27]"
TEMP/.venv/python TEMP/repo/scripts/verify_v27_clean_reproduction.py
TEMP/.venv/python TEMP/repo/scripts/audit_public_active_presidential_model_v27.py
```

## Result

- package installed as `election-forecast 0.27.0`
- frozen numerical environment: Windows, Python 3.13, NumPy 2.5.2, pandas 3.0.5,
  SciPy 1.18.1 and scikit-learn 1.9.0
- BLAS/OpenMP thread controls fixed to one worker
- V27 rebuilt in a separate temporary output directory
- Windows clean-clone reproduced `nested_predictions.csv` SHA-256:
  `f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b`
- frozen artifact hash: identical
- cross-platform CI additionally requires identical schema and categorical
  values and numeric agreement within absolute `1e-12`; it reports the rebuilt
  byte hash rather than assuming CSV serialization is platform-independent
- active V27 public audit: PASS
- regional development-panel macro MAE: `2.6139029869761212%p`
- national development-panel macro MAE: `0.7209938807856883%p`

GitHub CI also executes `scripts/verify_v27_clean_reproduction.py` from a clean
Windows checkout. Linux remains the general test platform, but is not claimed
as byte-identical: a diagnostic run found platform/BLAS drift beginning in the
2012 fold and reaching `0.001174501524589` in `conversion_resistance`. The
tagged commit must not be created until the canonical Windows reproduction and
the full Linux regression suite pass.
