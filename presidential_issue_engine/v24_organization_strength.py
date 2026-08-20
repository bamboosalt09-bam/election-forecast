"""Continuous non-major organisation strength from strictly prior PR evidence.

V23 assigns ``organization_strength`` from a five-value lookup on the bloc name
alone, so a three-seat splinter and a thirty-eight-seat third party receive the
same 0.60. This module replaces the non-major branch with a continuous value
built from the bloc's own share in the latest strictly earlier party-list
contest, expressed relative to the strongest major bloc in that same contest.

The single coefficient is anchored so that the 2017 제3지대 value reproduces the
legacy 0.60 exactly; it is a re-parameterisation of the existing constant, not a
new fitted quantity. Independents keep the legacy 0.15 floor and major blocs
keep 0.95, so the change is confined to party-backed non-major candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLOC_HISTORY = ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv"
PR_TYPES = frozenset(
    {"assembly_pr", "national_assembly_pr", "metro_council_pr", "local_council_pr"}
)
MAJOR = ("더불어민주당", "국민의힘")
ANCHOR = 0.5579  # reproduces the legacy 0.60 for pres_2017 제3지대
FLOOR = 0.15
CAP = 0.95

_PANEL: pd.DataFrame | None = None


def _panel() -> pd.DataFrame:
    global _PANEL
    if _PANEL is None:
        frame = pd.read_csv(BLOC_HISTORY, encoding="utf-8-sig")
        frame["yr"] = frame["election_id"].str.extract(r"(\d{4})").astype(int)
        _PANEL = frame.loc[frame["election_type"].isin(PR_TYPES)].copy()
    return _PANEL


def prior_relative_share(bloc: str, cutoff_year: float) -> float:
    """Bloc share in the latest strictly earlier PR contest, over that contest's major maximum."""

    prior = _panel().loc[_panel()["yr"] < cutoff_year]
    if prior.empty:
        return 0.0
    latest = prior.loc[prior["yr"].eq(prior["yr"].max())]
    shares = latest.groupby("bloc")["vote_share"].mean()
    ceiling = max(float(shares.get(name, 0.0)) for name in MAJOR)
    if ceiling <= 0.0:
        return 0.0
    return float(shares.get(bloc, 0.0)) / ceiling


def organization_strength(party_name: str, bloc: str, cutoff_year: float) -> float:
    """Return the V24 organisation strength for one candidate."""

    if "무소속" in str(party_name) or bloc == "무소속":
        return FLOOR
    if bloc in MAJOR:
        return CAP
    relative = prior_relative_share(bloc, cutoff_year)
    return float(np.clip(FLOOR + ANCHOR * relative, FLOOR, CAP))
