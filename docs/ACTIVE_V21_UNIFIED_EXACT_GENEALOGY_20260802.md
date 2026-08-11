# Active V21: Unified Exact Party Genealogy

## Decision

V21 replaces the split regional-party representation with one exact-party
lineage ledger. The same point-in-time estimator and routing formula is used in
every region and election family. V21 is promoted for representation
consistency, not because it improves retrospective MAE.

The active pointer is `data/config/current_presidential_model.json`. The
immediate rollback checkpoint is
`backups/model_checkpoints/20260802_pre_active_v21`.

## Canonical mechanism

1. Preserve the observed party name before any broad-camp normalization.
2. Replace collapsed Assembly constituency rows with exact NEC candidate-party
   rows and exclude independents from the party-terrain denominator.
3. Resolve an already-collapsed third-party label only when a strictly prior or
   same-date exact spatial profile passes the fixed similarity and margin
   rules. Otherwise retain `unresolved_third`.
4. Estimate candidate-ballot reliability from strictly prior same-date
   agreement with direct-party ballots. Direct-party ballots retain reliability
   1.0.
5. Fit regional lineage gaps using only events before the target election.
6. Project the same exact ledger to the legacy analytic bloc names only at the
   Ridge feature boundary. There is no second historical terrain source.
7. Route positive non-major lineage reservoirs with the same formula in all
   regions. Dated predecessor-successor paths may connect an inherited lineage
   to the candidate's exact party.

`source_quality` and lineage-resolution quality are deliberately separate. A
vote observation can be reliable as a broad third-party vote even when its
exact successor is uncertain. The former is used at the Ridge projection
boundary; the latter is used for exact-lineage identity.

## Dated genealogy facts

`data/raw/party_lineage_transitions.csv` stores factual rename and merger
events, not fitted vote multipliers. The initial registry contains:

- 국민중심당 -> 자유선진당, 2008-02-12, merger.
- 국민중심연합 -> 자유선진당, 2011-10-17, merger.
- 자유선진당 -> 선진통일당, 2012-05-29, rename.
- 선진통일당 -> 새누리당, 2012-11-16, merger.

Every graph edge has continuity and confidence 1.0 because the relation is a
completed factual rename or merger. These are not transfer-rate estimates. A
path is eligible only when every edge predates the target election.

The 2002 administrative-capital commitment is not a party genealogy event and
is therefore not encoded here. Treating that policy fact as party ancestry
would improve the historical fit but violate the measurement definition.

## Strict nested result

| Model | Regional macro MAE | National diagnostic MAE | Winner accuracy |
|---|---:|---:|---:|
| V20 | 3.217252%p | 1.481393%p | 80% |
| V21 | 3.395154%p | 1.765729%p | 80% |
| Change | +0.177902%p | +0.284336%p | 0%p |

| Election | V20 regional MAE | V21 regional MAE | Change |
|---|---:|---:|---:|
| 2002 | 3.636101%p | 4.084150%p | +0.448049%p |
| 2007 | 4.820084%p | 5.093073%p | +0.272989%p |
| 2012 | 2.194842%p | 2.459895%p | +0.265053%p |
| 2017 | 3.991872%p | 3.964735%p | -0.027137%p |
| 2022 | 1.443361%p | 1.373917%p | -0.069445%p |

The national metric uses observed regional contest votes and remains a
post-election aggregation diagnostic.

## Validation

- exact-lineage unit tests: 14 passed.
- full suite: 457 passed.
- strict PIT audit: PASS.
- all 10 active fold-audit rows exclude the target from fitting and preserve a
  consistent scored denominator.
- realized slot predictors and neutral-context direct adjustment are absent.
- V21 input-manifest paths containing 2025: 0.
- canonical and ablation nested-prediction SHA-256:
  `e9d5ca416868a790960f0337d8e9307d9bcc8ff1801c8b3e616ce6c41974ff57`.

## Artifacts

- `presidential_issue_engine/unified_lineage_identity.py`
- `data/raw/party_lineage_transitions.csv`
- `scripts/build_unified_exact_lineage_v21.py`
- `scripts/evaluate_unified_exact_lineage_v21.py`
- `scripts/run_active_presidential_model_v21.py`
- `data/config/active_presidential_model_v21.json`
- `outputs/unified_exact_lineage_v21/`
- `outputs/unified_exact_lineage_v21_ablation/`
- `outputs/active_presidential_nested_v21/`

## Remaining limitations

- Exact party labels already collapsed in an upstream historical file cannot
  always be recovered. They remain explicitly unresolved when the spatial
  evidence is insufficient.
- The dated transition registry is intentionally factual and incomplete. New
  edges require a dated source; they must not be inferred from target-election
  vote outcomes.
- Independent-candidate affinity is not party genealogy. It remains a separate
  candidate-history question and must not be forced into this graph.
