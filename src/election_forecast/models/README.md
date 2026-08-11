# Models

Model implementations consume feature tables and produce candidate-region-date predictions.

Planned modules:

- `utility_linear.py`: MVP linear Utility model.
- `ridge_vote_model.py`: leakage-safe ridge vote-share model based on the stats prototype.
- `ensemble.py`: model weighting and uncertainty summaries.

Rules:

- Models should not parse raw files directly.
- The public API should separate `fit`, `predict`, and `evaluate`.
- Backtests must report whether they are chronological or diagnostic cross-validation.
