# Active v17 automation tranche

> Superseded historical promotion record. The authoritative current model is
> V27; see `docs/FINAL_MODEL_V27_20260822.md`.

## Promotion

At this historical checkpoint the current model was v17, promoted from the v10 successor experiments. The
canonical runner is `scripts/run_current_presidential_model.py`; v16 remains a
byte-preserved rollback reference.

| Model | Regional weighted macro MAE | National candidate macro MAE | Winner accuracy |
|---|---:|---:|---:|
| active v16 | 3.3817 | 1.8417 | 0.80 |
| automatic response v10 | 3.3128 | 1.6185 | 0.80 |
| active v17 | **3.2133** | **1.4815** | **0.80** |

Active v17 combines:

1. footprint-controlled official candidate history v9;
2. target-excluded prior-only contest-response selection v10;
3. automatic 2007/2017 non-major Chungcheong routing;
4. a regional-party reservoir built from prior presidential, Assembly PR and
   district, local-council PR and district, and executive-election history;
5. coefficient-free rejection-beneficiary routing.

The v17 prediction CSV is byte-identical to the best strict nested v11
experiment. No post-2022 election row is present.

## Regional-party ablation

| Variant | Regional MAE | National MAE |
|---|---:|---:|
| v10 reference | 3.3128 | 1.6185 |
| automatic supplement, direct-party ballots | 3.2734 | 1.5790 |
| automatic supplement, full election history | **3.2133** | **1.4815** |
| automatic-only routing | 3.4161 | 1.7910 |

The automatic-only variant loses the 2002 policy-commitment and 2012
pre-election party-merger recipient facts. Election returns identify the size
of a regional-party reservoir but do not always identify its next recipient.
Those two factual rows remain until a dated event extractor replaces them.

## Third-candidate hold

The speech-only automatic third-candidate profile and source-lane pressure were
rerun on top of v17. Every automatic variant regressed because 2017 candidate
stature was understated.

| Variant | Regional MAE | National MAE | 2017 regional change |
|---|---:|---:|---:|
| v17 reference | **3.2133** | **1.4815** | 0.0000 |
| auto profile + manual pressure | 3.2682 | 1.7133 | +0.3729 |
| auto profile + no pressure | 3.3076 | 1.7459 | +0.6931 |
| auto profile + auto pressure | 3.3019 | 1.7526 | +0.6334 |

The manual third profile and pressure therefore remain active. Their next
replacement must use prior presidential competitiveness, party-list support,
district coverage, won offices, and withdrawal status, not speech stature
alone.

## Remaining automation plan

High feasibility:

1. third-candidate stature from prior candidate and party election history;
2. source-lane pressure from district-level party transition and ideological
   distance;
3. candidate political landscape cleanup using official candidate history;
4. 2002/2012 regional recipient facts from dated platform, merger, and
   endorsement records, with no manual numeric affinity.

External deterministic sources required:

1. generation weights: KOSIS age population plus NEC age turnout;
2. withdrawal facts: NEC registration status and dated coalition records;
3. issue taxonomy/intensity: pre-election Assembly salience, macro indicators,
   and a conservative factual event classifier.

Election outcomes must not be used to infer target-election issue intensity,
recipient affinity, or transfer compliance. These replacements are promoted
one lineage at a time only after strict nested all-election diagnostics.

## Recovery

- checkpoint: `backups/model_checkpoints/20260731_v10_automation_start/`
- v16 output: `outputs/active_presidential_nested_v16/`
- v17 output: `outputs/active_presidential_nested_v17/`
- active pointer: `data/config/current_presidential_model.json`
