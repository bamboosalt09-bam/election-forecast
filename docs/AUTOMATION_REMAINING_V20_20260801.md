# Remaining Automation After V20

Date: 2026-08-01

## Active status

V20 automates third-candidate viability, anti-major-party appeal, and regional
overlap for the eligible 2002, 2007, and 2017 rows. It does not yet automate
the full five-row profile. The current automation audit is
`outputs/automation_status_v5/`.

## High-feasibility next work

### 2012 final minor candidate

The active profile uses Kang Ji-won after the preliminary Ahn Cheol-soo
withdrawal lane is removed. The automatic profile source currently emits only
the political third-candidate rows available in the speech-derived context and
therefore omits Kang.

Required implementation:

1. seed every dated candidate from the official presidential candidate
   registry, including low-information candidates;
2. derive zero/low party organization from absence of prior direct-party and
   district evidence with explicit uncertainty rather than a hand-entered
   `0.02`;
3. keep preliminary withdrawn-candidate and final-ballot candidate identities
   as separate entities;
4. test target-election outcome mutation and no-evidence shrinkage.

### 2022 withdrawn Ahn Cheol-soo

The official candidate-history cache contains Ahn's 2017 presidential,
legislative, 2018 mayoral, 2021 withdrawal, and 2022 withdrawal records, but the
target candidate registry does not contain a separate `pres_2022` C row.

Required implementation:

1. build a dated preliminary-candidate registry independent of the final NEC
   ballot registry;
2. link person identity across target elections;
3. map People Party lineage to prior Assembly PR and district organization;
4. derive pre-withdrawal viability before applying a separate withdrawal
   transfer model;
5. exclude the 2022 result and every post-withdrawal observation from the
   pre-withdrawal feature cutoff.

### Party-level regional terrain

The V19 compiler successfully restores Liberal Democrats/Jaminryun lineage
records but did not improve both active diagnostics when inserted into the
presidential reservoir.

Next model:

- train party-preference magnitude on PR ballots;
- train organization persistence on district-versus-PR deviations within the
  same Assembly/local election;
- estimate candidate personal excess as candidate-ballot share minus the
  party terrain available before that election;
- use Assembly and local election pairs as the training objective;
- reserve presidential outcomes for downstream evaluation only.

This is preferable to selecting regional weights by presidential MAE over five
elections.

## Medium-feasibility work

### Centrist appeal

The automatic singleton failed because 2007 regional MAE regressed by
0.089345%p. Election results alone do not distinguish ideological centrality
from elite stature or broad public acceptability.

Use an outcome-independent latent distance model built from:

- Assembly political vectors;
- cross-party treatment and speech references;
- party coalition and merger graph;
- direct-party support breadth across regions;
- candidate office and party-leadership history.

The scale must be identified on Assembly/local party relationships, not on
presidential vote error.

### Third-candidate pressure

Automatic V16 pressure failed mainly because it produced nearly symmetric
2017 pressure against the two major candidates. Required inputs are:

- dated withdrawal or alliance event;
- candidate-to-camp ideological distance;
- party organization and direct-party support;
- two-major concentration before the event;
- recipient asymmetry from coalition and regime alignment.

Election returns can provide structural pressure but cannot identify a future
withdrawal event by themselves.

### Withdrawal transfer

Separate three quantities:

1. pre-withdrawal candidate support;
2. compliance/retention after withdrawal;
3. recipient distribution by ideological and coalition affinity.

The current presidential sample is too small to estimate all three from two
events. Use legislative/local alliance and withdrawal cases as external
training examples before promotion.

## Lower-feasibility work

- mega-issue taxonomy and shock class need dated event evidence in addition to
  Assembly salience;
- mega-issue polarity needs party defense/criticism attribution with a
  precision-first abstention policy;
- generation weights need KOSIS population and NEC age-turnout time series;
- campaign visibility and public attention require a dated media or search
  source, which is not currently active.

## Parameters that should not be automated

Safety caps, conservation bounds, minimum evidence counts, and VIF thresholds
are model-governance constraints. They should be documented and stress-tested,
not optimized against presidential MAE. The V5 audit therefore separates
14 safety bounds from the remaining behavioral parameters.

