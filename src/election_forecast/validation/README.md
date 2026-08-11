# Validation

Validation code checks model quality and guards against future-information leakage.

Planned modules:

- `rolling_origin.py`: train only on elections before the target election.
- `available_date.py`: enforce point-in-time filtering.
- `metrics.py`: MAE, RMSE, calibration summaries, and regional error reports.

Rules:

- Rolling-origin validation is the default headline metric.
- Leave-one-election-out can remain as a stability diagnostic, but it is not a historical forecast test.
- Reports must make leakage assumptions explicit.
