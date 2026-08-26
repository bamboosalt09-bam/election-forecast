<!-- active-model-version: v32 -->
# V31 software bill of materials

This is the human-readable SBOM required by Attachment 1 of the 2026 Open
Source Developer Competition result report. `requirements-v32.lock` records
the complete tested Python 3.13 runtime reproduction resolution for the active
model, including transitive packages, and is the file CI audits.
`requirements-v32.lock` resolves to the same 40 pinned packages as the V29 and
V30 locks - no version since has added or moved a dependency, and each
reproduction ran in the same environment. The files differ only in their header
comment, which names the version each was frozen for. They are copied rather
than aliased so each frozen version names its own lock. `requirements-v27.lock` is retained as the V27-era record; it
differs in `pandas` (3.0.3) and `pypdf` (6.16.1) and must not be used to
reproduce V31. Visualization is an optional extra; its two direct
packages are recorded below at the versions used for the published figures.

| Library | Tested version | License | Official repository | Purpose |
| --- | ---: | --- | --- | --- |
| NumPy | 2.4.6 | BSD-3-Clause | https://github.com/numpy/numpy | Arrays and deterministic numerical transforms |
| pandas | 3.0.5 | BSD-3-Clause | https://github.com/pandas-dev/pandas | Input joins, PIT filtering and output tables |
| SciPy | 1.18.0 | BSD-3-Clause | https://github.com/scipy/scipy | Scientific numerical routines used by the fitted stack |
| scikit-learn | 1.9.0 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | Ridge regression and preprocessing |
| Pydantic | 2.13.4 | MIT | https://github.com/pydantic/pydantic | Input schema validation |
| Beautiful Soup | 4.15.0 | MIT | https://git.launchpad.net/beautifulsoup | Public-page parsing utilities |
| feedparser | 6.0.14 | BSD-2-Clause | https://github.com/kurtmckee/feedparser | RSS/Atom source ingestion |
| HTTPX | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx | Official-source HTTP client |
| python-dateutil | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause | https://github.com/dateutil/dateutil | Date parsing and cutoff handling |
| Tenacity | 9.1.4 | Apache-2.0 | https://github.com/jd/tenacity | Bounded retry policy for collectors |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm | Local collection progress reporting |
| Typer | 0.27.1 | MIT | https://github.com/fastapi/typer | Auxiliary command-line interfaces |
| python-dotenv | 1.2.3 | BSD-3-Clause | https://github.com/theskumar/python-dotenv | Local-only environment configuration |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml | Configuration parsing |
| openpyxl | 3.1.5 | MIT | https://foss.heptapod.net/openpyxl/openpyxl | Spreadsheet import/export tools |
| pypdf | 6.16.2 | BSD-3-Clause | https://github.com/py-pdf/pypdf | Official-minute metadata extraction |
| matplotlib | 3.11.0 | PSF-based | https://github.com/matplotlib/matplotlib | Current figures |
| Shapely | 2.1.2 | BSD-3-Clause | https://github.com/shapely/shapely | Administrative-boundary dissolve for maps |

V31 exposes no external-model optional dependency extra. Historical stance
experiments, model weights and sentence corpora are excluded from the active
runtime and wheel.

**One model-derived table is the exception, and it is active.**
`data/raw/auto_issue_seed/candidate_issue_profile.csv` is a compact aggregate
produced with the open-weight encoder `jhgan/ko-sroberta-nli`; it ships, it is
read by the historical postprocess, and it is registered with its own rights
basis in `PUBLIC_DATA_SOURCES.json` rather than under project authorship. It
contains no model weight and no source sentence. Removing it is not free: a
full-removal diagnostic moved regional macro MAE from `2.613903%p` to
`4.935929%p` and winner accuracy from `0.8` to `0.6`. Saying model-derived
tables are excluded without naming this one would be false.

## Reproduction environment

```bash
python -m pip install -r requirements-v32.lock
python -m pip install --no-deps .
election-forecast audit-current-presidential
election-forecast verify-current-presidential
```

Package metadata remains the source of supported dependency ranges. The lock
is the reviewed exact environment for the frozen V31 reproduction check.
