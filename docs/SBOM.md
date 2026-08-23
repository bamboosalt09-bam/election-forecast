# V29 software bill of materials

This is the human-readable SBOM required by Attachment 1 of the 2026 Open
Source Developer Competition result report. `requirements-v27.lock` records
the complete tested Python 3.13 runtime reproduction resolution, including
transitive packages. Visualization is an optional extra; its two direct
packages are recorded below at the versions used for the published figures.

| Library | Tested version | License | Official repository | Purpose |
| --- | ---: | --- | --- | --- |
| NumPy | 2.5.2 | BSD-3-Clause | https://github.com/numpy/numpy | Arrays and deterministic numerical transforms |
| pandas | 3.0.5 | BSD-3-Clause | https://github.com/pandas-dev/pandas | Input joins, PIT filtering and output tables |
| SciPy | 1.18.1 | BSD-3-Clause | https://github.com/scipy/scipy | Scientific numerical routines used by the fitted stack |
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
| pypdf | 6.16.1 | BSD-3-Clause | https://github.com/py-pdf/pypdf | Official-minute metadata extraction |
| matplotlib | 3.11.0 | PSF-based | https://github.com/matplotlib/matplotlib | Current figures |
| Shapely | 2.1.2 | BSD-3-Clause | https://github.com/shapely/shapely | Administrative-boundary dissolve for maps |

V29 exposes no external-model optional dependency extra. Historical stance
experiments, weights and model-derived tables are excluded from the active
runtime and wheel.

## Reproduction environment

```bash
python -m pip install -r requirements-v27.lock
python -m pip install --no-deps .
election-forecast audit-current-presidential
election-forecast verify-current-presidential
```

Package metadata remains the source of supported dependency ranges. The lock
is the reviewed exact environment for the frozen V27 reproduction check.
