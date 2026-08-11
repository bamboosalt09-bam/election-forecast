"""Election type / contest / aggregation rule — the cross-election seam.

The key structural insight behind the open-source roadmap is that a presidential
race is NOT simpler than a legislative race; it is the special case where many
regional A/B/C/alpha contests are aggregated into one national vote share. The
other election types reuse the same slot machinery but change the
*aggregation rule* that turns slot utilities into an outcome.

| election_type | typical contest unit        | aggregation_rule        | outcome unit       |
|---------------|-----------------------------|-------------------------|--------------------|
| presidential  | one national contest        | national_vote_share     | nationwide % share |
| legislative   | one per electoral district  | district_winner + seats | seat counts        |
| local         | one per office x district   | per-office              | office winners     |

The statistics competition only ever emits `presidential` rows, so its data
carries these columns as constants. The open-source engine reads them to pick
the right aggregation module.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel


class ElectionType(str, Enum):
    PRESIDENTIAL = "presidential"
    LEGISLATIVE = "legislative"  # 총선 — open-source target, not stats scope
    LOCAL = "local"  # 지선 — open-source target, not stats scope


class AggregationRule(str, Enum):
    """How slot utilities become an outcome for a contest."""

    NATIONAL_VOTE_SHARE = "national_vote_share"  # presidential: sum regions -> nation
    DISTRICT_WINNER = "district_winner"  # legislative single-member: top slot wins
    SEAT_ALLOCATION = "seat_allocation"  # legislative proportional: shares -> seats
    MULTI_MEMBER = "multi_member"  # local councils: top-N slots win


ELECTION_TYPES = tuple(e.value for e in ElectionType)
AGGREGATION_RULES = tuple(a.value for a in AggregationRule)


class RegionResolution(str, Enum):
    """Spatial granularity a dataset is keyed on.

    Diverges by project: the statistics competition uses ``sido`` (17 metro/
    province units, boundary-stable across the 2002–2025 panel); the open-source
    project targets ``sigungu`` (~250 districts), which legislative/local
    elections require anyway. Recorded so cross-election joins never silently mix
    incompatible region keys.
    """

    SIDO = "sido"          # 시도/광역 (17) — stable over 20 years; stats default
    SIGUNGU = "sigungu"    # 시군구 (~250) — oss target; needed for 총선/지선
    DISTRICT = "district"  # 선거구 (legislative) — oss, future


REGION_RESOLUTIONS = tuple(r.value for r in RegionResolution)


# Which aggregation rules are meaningful for each election type. The engine uses
# this for validation; the stats project never leaves the presidential row.
_DEFAULT_RULES = {
    ElectionType.PRESIDENTIAL: AggregationRule.NATIONAL_VOTE_SHARE,
    ElectionType.LEGISLATIVE: AggregationRule.DISTRICT_WINNER,
    ElectionType.LOCAL: AggregationRule.MULTI_MEMBER,
}


def default_aggregation_rule(election_type: str) -> str:
    """Return the canonical aggregation rule for an election type."""

    return _DEFAULT_RULES[ElectionType(election_type)].value


class ContestRow(BaseModel):
    """One race within an election.

    Presidential elections have exactly one contest. Legislative and local
    elections have many. `region_ids` lists the regions whose slot utilities are
    aggregated for this contest (a single national list for presidential, a
    single district for legislative, etc.).
    """

    election_id: str
    election_type: str = ElectionType.PRESIDENTIAL.value
    contest_id: str
    contest_name: str
    aggregation_rule: str = AggregationRule.NATIONAL_VOTE_SHARE.value
    seats: int = 1  # >1 only for proportional / multi-member contests
    notes: str | None = None


def presidential_contest_defaults(election_id: str, region_ids: List[str]) -> ContestRow:
    """Build the single all-regions contest a presidential election uses.

    Convenience for the statistics competition so it never has to author a
    contests file by hand: one national contest covering every region.
    """

    return ContestRow(
        election_id=election_id,
        election_type=ElectionType.PRESIDENTIAL.value,
        contest_id=f"{election_id}__national",
        contest_name="National",
        aggregation_rule=AggregationRule.NATIONAL_VOTE_SHARE.value,
        seats=1,
        notes=f"covers {len(region_ids)} regions",
    )
