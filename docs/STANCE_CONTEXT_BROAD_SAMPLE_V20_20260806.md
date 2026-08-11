# Stance Context Broad-Sample V20 Check (2026-08-06)

## Scope

This check expanded sentence coverage without changing the active forecast or
the frozen V20 stance rules.

- source: frozen 15th-21st Assembly extraction;
- elections: `pres_2002` through `pres_2022` only;
- vote outcomes: not read;
- 2025 rows: not used;
- active forecast V23: unchanged.

The source CSV was read only. No Assembly PDF/XLSX extraction was rerun.

## Sampling change

The previous fresh E/F samples contained 10,000 rows, with 1,000 rows per
election in each sample and an 80% directional-cue quota. Their combined target
mix was heavily government-oriented.

The new broad corpus contains 25,000 unique, previously unused sentences. It
covers all seven Assemblies, all five scored elections, all three target types,
19 issue groups, and 579 election-Assembly-target-issue cells. Each election
contributes 5,000 rows. Half of the corpus is deterministic representative
min-hash sampling; the other half broadens rare targets and coverage cells.

Because historical person-target extraction is sparse, fixed target quotas
cannot be met in every election. The selector uses all available rare-target
rows instead of duplicating sentences. The final 25,000-row target counts are:

- government: 15,647;
- party: 5,053;
- person: 4,300.

A 10,000-row analysis slice was then selected for immediate NLI inference:
5,000 representative rows and 5,000 coverage rows. It contains 2,000 rows per
election and retains all seven Assemblies and all 19 issues.

## V20 distribution

| Cohort | Positive | Negative | Neutral | Directional rate |
|---|---:|---:|---:|---:|
| Previous cue-rich E/F | 1 | 88 | 9,911 | 0.890% |
| New broad analysis | 0 | 33 | 9,967 | 0.330% |
| New representative half | 0 | 11 | 4,989 | 0.220% |
| New coverage half | 0 | 22 | 4,978 | 0.440% |
| Combined 20,000 | 1 | 121 | 19,878 | 0.610% |

The combined positive:negative:neutral ratio is therefore
`1:121:19,878`. The representative new half is the least selection-biased
estimate and yields `0:11:4,989`.

Before the V20 strict-owner gate, the new broad 10,000 rows contained eight
positive and 44 negative predictions. V20 neutralized all eight positives and
11 negatives. This confirms that the near-absence of positive labels is mainly
a consequence of the explicit first-person positive ownership rule, not proof
that positive parliamentary language is absent.

## Coverage result

Directional emissions occur in every election and in the 15th-20th
Assemblies. The 21st Assembly has zero V20 directional emissions in the 813-row
slice. Person targets emit direction most often (`18/1,842`, 0.977%);
government and party targets emit about 0.18-0.20%.

Housing, external shock, candidate competence, and inflation/livelihood have
the highest directional rates. Nine issue groups have no V20 directional
emission in this slice.

## Interpretation and decision

The old `1:88:9,911` ratio was not representative of the full target-bearing
corpus because the old selector forced 80% cue-rich rows. Broadening the sample
reduces observed V20 directional coverage from 0.89% to 0.33%, and to 0.22% in
the representative half.

The 33 new emissions are not an independent locked precision audit. A content
inspection found that most are genuine explicit criticism, but it also exposed
remaining target-owner failures. For example, a sentence praises Hannara's
warning and criticizes the government, while the extracted row target is
Hannara; V20 preserves the negative label against the wrong target. A second
row attributes an economic crisis to a prior government while mentioning that
Hannara cannot deny it. These are precisely the unresolved discourse-target
problem identified in the V20 audit.

Therefore:

1. V20 remains shadow-only and is not promoted.
2. The active forecast and its metrics remain unchanged.
3. Lowering the neutral threshold is not justified.
4. The next classifier must resolve stance owner and target before polarity.
5. A future confirmatory audit must use new locked rows; this broad sample is
   now development evidence.

## Artifacts

- `data/shadow/stance_context_broad_25000_v1/stance_context_broad_25000.csv`
- `data/shadow/stance_context_broad_25000_v1/stance_context_broad_analysis_10000.csv`
- `outputs/assembly_stance/stance_context_strict_owner_v20/broad_analysis_10000_base`
- `outputs/assembly_stance/stance_context_strict_owner_v20/broad_analysis_10000_v20`
- `broad_v20_distribution.json`
- `broad_v20_group_distribution.csv`
- `directional_emissions_33.csv`

The chunked encoder stores one checkpoint per 1,000 rows and can resume after
interruption without recomputing completed chunks.

Metadata, scripts, diagnostics, and this record are backed up at
`backups/model_checkpoints/20260806_stance_context_broad_sample_v20`.
