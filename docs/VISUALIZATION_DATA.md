# Visualization data and licensing

`presidential_issue_engine/make_poster_figures.py` now generates only current
V28 figures. Retrospective charts read the finalized through-2022 development
panel. Forecast charts read the separate `pres_2025` V28 D-1 artifact and never
read the realized 2025 result.

The old numbered PNGs were moved to
`research/visualizations/legacy_v23_v25/`. They are superseded V23/V25 research
history, are excluded from the packaged runtime and must not be cited as V28.

The current public set adds a V28 overview and an execution architecture figure
to the performance, election-level, regional and 2025 D-1 map views. Every
current filename starts with `v28_` and every chart reads only the declared V28
historical or corrected-demonstration artifact.

## Map source

- Dataset: `vuski/admdongkor`, `ver20250401/HangJeongDong_ver20250401.geojson`
- Basis: Statistics Korea SGIS census administrative boundaries, corrected and
  time-versioned by the `admdongkor` project
- Snapshot: 2025-04-01, before the 2025-06-02 forecast cutoff
- Source commit: `fbcac3db020609dce5831a856a6d5aa5cb40a908`
- Pinned SHA-256: `1b80c423c82a9349859aef020174c1276896943d064762fc2e184f75f5ee2ceb`
- License: CC BY 4.0 with Korea Open Government License Type 1 attribution
- Source: `https://github.com/vuski/admdongkor`
- Terms: `https://github.com/vuski/admdongkor/blob/master/LICENSE-DATA`

The generator dissolves 3,554 administrative-dong features into 17 first-level
regions using the dated source code. This replaces the former Natural Earth
world-scale generalized layer. Hash drift or missing regions abort generation.

Every map circle has a fixed radius; size does not encode population. Its
sectors sum to 100% within that region.

## Reproduction

```bash
pip install -e ".[viz]"
python presidential_issue_engine/make_poster_figures.py
```

Map generation requires network access to the hash-pinned snapshot.
