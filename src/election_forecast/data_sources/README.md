# Data Sources

Adapters in this package convert raw public files into stable project schemas.

Planned modules:

- `nec_proportional.py`: NEC National Assembly proportional and local-council proportional results.
- `nec_presidential.py`: standardized presidential election results.
- `polls.py`: poll ingestion with `published_date` and `available_date`.
- `news.py`: manually scored or semi-automated issue/news inputs.

Rules:

- Raw source parsing belongs here, not in model code.
- Every emitted row must have an explicit event date or election date when relevant.
- Inputs used for backtests must also expose `available_date` when the source can be known after the event.
