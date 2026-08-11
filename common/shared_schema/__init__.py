"""Election-type, contest, and aggregation-rule dimensions.

These three dimensions are the "room" the schema leaves for the open-source
project to grow from presidential to legislative (총선) and local (지선)
elections WITHOUT a schema rewrite. The statistics competition fixes all three
to their presidential defaults and never touches the extra complexity.
"""

from common.shared_schema.election import (
    AGGREGATION_RULES,
    ELECTION_TYPES,
    REGION_RESOLUTIONS,
    AggregationRule,
    ContestRow,
    ElectionType,
    RegionResolution,
    default_aggregation_rule,
    presidential_contest_defaults,
)

__all__ = [
    "ELECTION_TYPES",
    "AGGREGATION_RULES",
    "REGION_RESOLUTIONS",
    "AggregationRule",
    "ElectionType",
    "RegionResolution",
    "ContestRow",
    "default_aggregation_rule",
    "presidential_contest_defaults",
]
