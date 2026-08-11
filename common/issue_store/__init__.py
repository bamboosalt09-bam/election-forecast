"""Issue/event store — the populator-agnostic "issue memory".

The core abstraction behind "이슈를 모아 메모리에 저장하고 분석한다": a structured,
dated record of issues/events that the forecast consumes. *How* the store gets
filled is a pluggable populator (mirrors the swappable-scorer principle):

- ``curated``  : a human writes issue/event rows directly (stats competition, fast, AI-free).
- ``aggregate``: filled from BIGKinds issue analysis + search-trend salience (no body analysis).
- ``corpus``   : mined from hundreds of thousands of article bodies via news_analyzer (OSS).

All three emit the SAME :class:`IssueEventRow`, and the SAME :func:`rollup` turns
the store into ``common.feature_schema`` variables. So the statistics competition
and the open-source pipeline share one issue memory and one rollup.
"""

from common.issue_store.schema import (
    ISSUE_TYPES,
    POPULATORS,
    IssueEventRow,
    issue_columns,
    validate_issue_frame,
)
from common.issue_store.rollup import rollup_issue_features

__all__ = [
    "ISSUE_TYPES",
    "POPULATORS",
    "IssueEventRow",
    "issue_columns",
    "validate_issue_frame",
    "rollup_issue_features",
]
