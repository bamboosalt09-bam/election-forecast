# KOSPI Official-Source Migration Audit

## Status

- GitHub publication may proceed with the source boundary below.
- The active V23 input has not been replaced.
- The commercial/user-provided KOSPI export is retained locally but excluded
  from Git tracking.
- An official-source candidate was fetched from Bank of Korea ECOS and stored
  in the ignored local official-source cache.

## Official source

- Provider: Bank of Korea Economic Statistics System (ECOS)
- API: `StatisticSearch`
- Table: `802Y001` (`1.5.1.1. Stock market, daily`)
- Item: `0001000` (`KOSPI index`)
- Frequency: daily
- Original source organization reported by ECOS: Korea Exchange
- Model boundary: `1995-01-03` through `2022-12-31`
- Observed rows: 7,112
- Last trading observation in the requested range: `2022-12-29`

The public repository contains the fetcher and provenance manifest. The
downloaded row-level CSV remains under `data/raw/official_sources/cache/`,
which is ignored, unless redistribution permission is confirmed separately.

## Comparison with the legacy local file

The official ECOS candidate and the legacy file overlap on 7,097 dates.

- exact closing-index matches: 7,090
- nonmatching closes: 7
- mean absolute close difference on overlap: 0.000323 index points
- maximum absolute close difference: 1.41 index points
- ECOS-only trading dates: 15
- legacy-only dates inside ECOS coverage: 1

For the 2002, 2007, 2012, 2017, and 2022 presidential forecasts, the latest
pre-election close, 3-month return, 12-month return, and 12-month drawdown are
identical. The market-stress index changes by at most 0.037739 because the
official daily series begins in 1995 instead of 1990. The resulting assembled
feature changes are small:

- `issue_advantage`: maximum absolute change `0.000011214`
- `rif`: maximum absolute change `0.000003478`
- partisan and political-landscape predictors: unchanged

## Promotion rule

Do not overwrite frozen V23 artifacts in place. To activate the official
series:

1. create a new versioned source-migration run;
2. point that run to the ECOS cache file;
3. rerun strict nested predictions and all point-in-time audits;
4. compare predictions and metrics with frozen V23;
5. promote only after input manifests and documentation use the new hashes.

This is a provenance migration, not a reason to tune model parameters.
