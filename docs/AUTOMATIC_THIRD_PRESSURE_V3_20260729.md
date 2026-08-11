# Automatic third-candidate pressure v3

Date: 2026-07-29

## Purpose

This experiment tests whether `data/raw/third_candidate_pressure.csv` can be
replaced without using presidential outcomes. Active v16 and candidate v2 are
unchanged.

## Automatic definition

For each active non-major candidate, total draw propensity is:

`sqrt(centrist appeal * anti-major-party appeal)`

The total is allocated across the two major-party source lanes using normalized
affinity. Lane affinity is the equal mean of:

1. political-axis affinity to the source bloc
2. centrist appeal
3. anti-major-party appeal

The downstream engine already multiplies pressure by automatic third-candidate
viability, so viability is not counted again. Confidence is the geometric mean
of profile, speech, source-candidate, and landscape confidence. There are no
election-specific pressure constants.

## Generated pressure

| Election | Third candidate | Liberal-source pressure | Conservative-source pressure |
|---|---|---:|---:|
| 2002 | Kwon Young-ghil | 0.174 | 0.057 |
| 2007 | Lee Hoi-chang | 0.178 | 0.221 |
| 2017 | Ahn Cheol-soo | 0.227 | 0.237 |

The source-slot letters differ by election; the table names the actual source
bloc to avoid slot ambiguity.

## Strict nested ablation

| Pressure input | Regional MAE | National MAE | Winner accuracy |
|---|---:|---:|---:|
| Manual v2 baseline | **3.335%p** | **1.781%p** | 80% |
| No pressure | 3.366%p | 1.812%p | 80% |
| Automatic pressure | 3.379%p | 1.846%p | 80% |

Automatic minus manual national MAE by affected election:

| Election | Difference |
|---|---:|
| 2002 | +0.137%p |
| 2007 | -0.153%p |
| 2017 | +0.342%p |

The automatic formula improves 2007 but regresses 2002 and 2017. The no-pressure
model is only about 0.031%p worse than the manual baseline at both macro levels,
which shows that the active model is not heavily dependent on this file.

## Diagnosis

The Assembly-derived candidate political axes are too flat to recover the 2017
source split. Ahn Cheol-soo is nearly equidistant from the two source lanes in
those signals. His completed pre-2017 Assembly party history also points toward
the liberal lineage, while the observed conservative defection was conditional
on the impeachment rupture. Making the automatic conservative pressure large
only because that improves 2017 would tune against the target outcome.

The missing structure is an interaction between candidate affinity and a
source camp's pre-election rupture or burden. It should be developed as a
general contest-regime interaction, not as a replacement number for 2017.

## Leakage and lineage

- full test suite: `410 passed`
- manual pressure reads in automatic variants: zero
- post-2022 outcomes: absent
- target excluded from every outer fit: yes
- realized-slot predictors: absent
- new-layer outcome fields: none
- generated pressure path is recorded in the run input manifest with SHA-256

## Decision

Do not promote automatic pressure v3. Preserve it as a rejected but informative
experiment. Keep active v16 and candidate v2 unchanged. The next defensible
work is to build a general affinity-by-source-rupture interaction and test it
without selecting its strength from the same presidential outcomes.
