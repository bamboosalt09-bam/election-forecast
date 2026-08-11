# Official-data automation roadmap (through 2022 only)

## Objective

Replace election-specific manual strengths with dated factual inputs and one
universal compiler. The active v16 model remains frozen until a replacement
passes strict nested, leakage, provenance, and election-level regression checks.

The automation boundary is fixed at 2022-12-31. No 2025 outcome, candidate
record, comparison metric, or tuning result may enter persisted inputs or model
selection.

## Official source map

| Domain | Official source | Automation use | Current status |
|---|---|---|---|
| Candidate identity and history | NEC candidate integrated search, `CndaSrchService/getCndaSrchInqire` | Birthday-based entity resolution, prior candidacies, party, district, career, registration status | Implemented and collected |
| Candidate roster | NEC candidate information, `PofelcddInfoInqireService/getPofelcddRegistSttusInfoInqire` | Replace result-file-derived candidate lists with registration-time rosters | Collector implemented; current key receives HTTP 403 for this service |
| Withdrawal/death/invalidation | NEC `CndaRegInvdInqireService/getCndaRsgtDthInvdInqire` | Dated event fact for withdrawal handling | Endpoint verified; service approval and event-date schema still required |
| Election and party terrain | NEC vote/count API and existing official result files | Strictly prior party/direct-ballot regional terrain | Already active through `bloc_history_results.csv` |
| Assembly office and district history | NEC candidate history plus National Assembly member API | Candidate office scope, repeat tenure, constituency history | NEC component implemented; Assembly replacement API mapping pending |
| Age structure and turnout | KOSIS OpenAPI plus NEC age-turnout analysis files | Election-generation weights and turnout-sensitive cohort mass | Source verified; collector not yet implemented |
| Issue salience and direction | National Assembly transcripts, dated official event facts, NEC party/candidate policy APIs | Automatic taxonomy, intensity, persistence, target attribution | Transcript layer exists; fully automatic taxonomy/intensity remains incomplete |

Official references:

- Candidate integrated search: https://www.data.go.kr/data/15140045/openapi.do
- Candidate information: https://www.data.go.kr/data/15000908/openapi.do
- Candidate withdrawal/invalidation: https://www.data.go.kr/data/15111382/openapi.do
- Vote/count information: https://www.data.go.kr/data/15000900/openapi.do
- KOSIS OpenAPI: https://kosis.kr/openapi/index/index.jsp
- National Assembly API catalogue: https://www.data.go.kr/data/15125891/openapi.do

## Implemented in v1

### Credential-safe API client

`src/news_collector/sources/public_data_api.py`

- environment-only service key;
- paginated fetch;
- retry only for timeout, transport, HTTP 429, and 5xx;
- immediate stop for unauthorized 4xx;
- atomic cache writes;
- SHA-256 response provenance;
- cache identity and metadata exclude the service key;
- offline replay from cache.

### Candidate history collector

`scripts/collect_official_candidate_history.py`

- reads only `election_id,slot,candidate_name,party_name` from the existing
  reference table;
- queries 13 unique candidate names for the 15 target candidate/election rows;
- discards records after 2022-12-31 in memory before checkpoint writes;
- resolves same-name people with target election, party, and birthday;
- leaves ambiguous identities unresolved instead of guessing;
- masks target-election win/loss and permits win/loss only from strictly prior
  elections;
- writes restartable per-name sanitized checkpoints.

Current factual result:

- 15/15 candidate-election identities resolved;
- 77 candidate-history rows through 2022;
- 13 candidate-region evidence rows;
- no persisted `2025`, `20250603`, `serviceKey`, or provided key prefix;
- active model not changed.

### Candidate regional compiler

`presidential_issue_engine/official_candidate_history.py`

The compiler uses only records with:

```text
source_election_date < target_election_date
```

For eligible non-national offices:

```text
record_evidence
  = office_scope_weight
  * recency_decay
  * strictly_prior_win_weight
  * entity_match_confidence
```

Aggregated evidence is converted through bounded saturation functions to
`regional_affinity` and `organization_depth`. This is a candidate electoral
footprint, not yet a complete political-regional identity measure.

## Strict nested ablation

Output: `outputs/official_candidate_regional_base_v5_ablation/`

| Variant | Regional weighted MAE | National candidate MAE | Winner accuracy |
|---|---:|---:|---:|
| Active manual v16 | 3.3817%p | 1.8417%p | 0.80 |
| Official history only | 3.5865%p | 2.0693%p | 0.60 |
| Official history + automatic prior-party organization | 3.5885%p | 2.0691%p | 0.60 |

Decision: **not promoted**.

The official records are valid facts, but the first compiler over-interprets
ordinary constituency service as durable personal regionalism. It also cannot
by itself recover party-organization bases such as a third-party Honam base,
movement bases, birthplace identity, or spillover across a broader region. The
data collector is retained; the strength compiler remains experimental.

## Remaining manual inputs and replacement rule

| Current input | Automatic replacement |
|---|---|
| `candidate_regional_base.csv` | Official office/candidacy facts + strictly prior party organization + repeated/high-office gate + weak single-district cap |
| `chungcheong_identity_alignment.csv` | Candidate-base evidence routed through prior regional third-bloc excess; no election-specific strength |
| `third_candidate_profile.csv` | Preliminary expected share, major-party status, organization, office history, and speech-derived political vector |
| `third_candidate_pressure.csv` | Same-lane ideological distance and preliminary viability; no realized vote rank |
| `withdrawal_event_profiles.csv` | Official dated withdrawal fact, event timing, formal endorsement, and pre-event candidate strength |
| `withdrawn_candidate_transfers.csv` | Universal transfer model from ideological affinity, voter compliance prior, timing, and target endorsement |
| `mega_issue_intensity.csv` | Dated transcript salience burst, speaker influence, breadth, persistence, and official event severity |
| `mega_issue_taxonomy.csv` | Rule/ML hybrid event classifier with conservative abstention and source-level provenance |
| `election_generation_weights.csv` | KOSIS population by age x historical NEC age-turnout rates, each released before target |
| `candidate_political_landscape.csv` | Assembly speech vectors, party affiliation, office history, and external-party treatment with PIT filters |

## Fixed numeric parameters

The 49 inventoried numeric constants must be split into two classes:

1. Safety parameters: caps, floors, and numerical stability constants. These may
   remain universal and documented.
2. Behavioral parameters: gains, half-lives, transfer rates, and activation
   thresholds. These must be fixed from theory/external evidence or selected
   inside prior-only inner folds and frozen before the target fold.

No election-specific numeric strength may be copied into deployment inputs.

## Next implementation order

1. Obtain service approval for NEC candidate information and withdrawal APIs;
   run `collect_official_presidential_registry.py` and add dated withdrawal facts.
2. Replace the first regional compiler with a multi-source gated model:
   high office/repetition can create personal base; a single constituency can
   only create a weak footprint; party organization remains a separate signal.
3. Add KOSIS age-population and historical NEC age-turnout collectors with
   release-date metadata and fold-specific availability checks.
4. Build automatic mega-issue taxonomy/intensity from transcript burst,
   cross-party breadth, speaker office/seniority, event class, and residual
   persistence. Direction and intensity must remain separate.
5. Convert withdrawal transfer strengths and third-candidate character to
   formulas based on pre-election candidate features.
6. Run strict nested ablations one layer at a time. Promotion requires no
   provenance violation, aggregate improvement, and no concentrated election
   regression that reverses winners.

## Commands

```powershell
$env:DATA_GO_KR_SERVICE_KEY = "..."
python scripts/collect_official_candidate_history.py
python scripts/collect_official_presidential_registry.py
python scripts/evaluate_official_candidate_regional_base_v5.py
python -m pytest -q
```

The candidate-history command can be rerun with `--offline`; completed names are
read from sanitized checkpoints. The registry command currently requires the
candidate-information service to be added to the key's data.go.kr approvals.
