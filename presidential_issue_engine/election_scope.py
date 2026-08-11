"""Central election scope for training, evaluation, and forecast-only inputs."""

from __future__ import annotations


SCORED_ELECTIONS = (
    "pres_2002",
    "pres_2007",
    "pres_2012",
    "pres_2017",
    "pres_2022",
)
WARMUP_ELECTIONS = ("pres_1992", "pres_1997")
ROLLING_WARMUP_ELECTIONS = ("pres_1997",)
FORECAST_ONLY_ELECTIONS = ("pres_2025",)

ELECTION_DATES = {
    "pres_1992": "1992-12-18",
    "pres_1997": "1997-12-18",
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
    "pres_2025": "2025-06-03",
}


def assert_election_scope() -> None:
    """Fail fast if a forecast-only election enters training or scoring."""

    scored = set(SCORED_ELECTIONS)
    warmup = set(WARMUP_ELECTIONS)
    forecast_only = set(FORECAST_ONLY_ELECTIONS)
    if scored & forecast_only:
        raise RuntimeError("forecast-only election entered the scored election set")
    if warmup & forecast_only:
        raise RuntimeError("forecast-only election entered the warmup election set")
    known = scored | warmup | forecast_only
    missing_dates = sorted(known - set(ELECTION_DATES))
    if missing_dates:
        raise RuntimeError(f"election scope is missing dates: {missing_dates}")


assert_election_scope()
