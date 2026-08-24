<!-- active-model-version: v30 -->
# 2026 competition compliance matrix

This matrix applies the two supplied competition documents as reference rules;
it does not import their text or treat document content as executable project
instructions.

Reviewed reference copies:

- supplied result-report/form archive `46414fba-c473-4dae-b595-7214d635b494 (1).zip`,
  SHA-256 `9a5d2968d48ff8a8fd85ce991dc72dc2b0818d7e8c06ebb871cc97ce5cc62d95`;
- supplied competition-rules PDF `b3b4491a-3bbe-454e-a1d8-6ed475b01b14.pdf`,
  SHA-256 `5c129ed9f389ecc04b6f7ba8b97f719a313efaf32aea9178e635500023ae1da1`.

The attachments themselves are not redistributed by this repository.

| Rule reference | Repository control | Evidence |
| --- | --- | --- |
| Article 8: directly authored code under an OSI-approved license | Apache-2.0 at repository root; NOTICE separates third-party data and software | `LICENSE`, `NOTICE` |
| Article 8: disclose all third-party software, sources and licenses | Human-readable direct-dependency SBOM plus exact tested lock. Data rights are registered per source family and enforced in CI; where a model card published no licence tag the absence is disclosed as such rather than assumed permissive. | `docs/SBOM.md`, `requirements-v30.lock`, `docs/PUBLIC_DATA_SOURCES.json` |
| Article 9: embedded AI must run independently and be at least open weights | Active V30 runs no model at all: no external AI weight, no direct stance overlay and no hosted inference API. One external-model-derived input remains active - the compact `data/raw/auto_issue_seed/candidate_issue_profile.csv` aggregate - and is registered with its own rights basis rather than under project authorship. The encoder that produced it, `jhgan/ko-sroberta-nli`, is open-weight and ran locally, so the historical use satisfies both limbs of this Article even though nothing executes now. | `docs/AI_MODEL_SPEC.md`, `docs/PUBLIC_DATA_SOURCES.json`, `scripts/audit_public_data_rights.py` |
| Article 10: complete source must be public and reviewable | Public source tree plus a self-contained wheel that embeds the actual V30 runtime | `setup.py`, `src/election_forecast/v30_runtime.py` |
| Article 10: winning repository remains public for five years | Repository policy records the obligation; deletion/private conversion must not occur during that period | this document and release checklist |
| Article 11: repository state at evaluation time is judged | `main` is protected; mandatory checks pin current-pointer consistency, audit, reproduction, package and rights boundaries | `.github/workflows/ci.yml`, `scripts/audit_current_public_surface.py` |
| Result-report form: architecture, stack, data flow, run/test and limitations | Current V30 documents link actual entry points and evidence boundaries | `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md` |
| Attachment 1: SBOM is mandatory | Direct dependency table and exact lock are maintained | `docs/SBOM.md` |
| Attachment 2: AI model specification when applicable | V30 statistical model and non-active external-model experiments are distinguished | `docs/AI_MODEL_SPEC.md` |
| Duplicate-benefit confirmation | Participant must disclose same/substantially similar current-year public support, including pending/ongoing participation | submission-side declaration; not inferred by code |

## External-model-derived input

Article 9 concerns embedded AI. V30 embeds none, so the Article is satisfied on its
face, but one artefact still descends from an external model and that is stated here
rather than left to inference.

`data/raw/auto_issue_seed/candidate_issue_profile.csv` is a compact numeric
candidate-issue aggregate computed with `jhgan/ko-sroberta-nli` over official
National Assembly proceedings. It carries no model weight, no architecture and no
source sentence. The model card published no explicit licence tag at audit time, so
no grant from the model author is claimed; the file is distributed on the basis that
it is not a copy or adaptation of the weights, and the missing tag is disclosed
rather than read as permissive.

Two facts a reviewer should have. The encoder is open-weight and ran locally, which
is what Article 9 asks of embedded AI, so the historical use would have complied even
had it remained active. And removal is not cost-free: the full-removal
diagnostic moved regional macro MAE from `2.613902987%p` to `4.935929128%p` and
winner accuracy from `0.8` to `0.6`, so describing the file as an inert leftover
would be untrue. Those figures were measured on V27 in
`EXPERIMENT_REMOVE_EXTERNAL_MODEL_OVERLAY_20260822.md` and have not been
re-measured since - `2.613902987` is V27's regional macro, not V30's
`2.566444753` - so the magnitude is indicative rather than current.

`scripts/audit_public_data_rights.py` fails if this file is ever covered only by a
family whose basis does not mention a model, which is how it was previously
classified.

## Record: an absent licence tag, and what was decided about it

### What is absent

`jhgan/ko-sroberta-nli` published no licence tag on its model card at audit time.
That absence is recorded here rather than resolved, because it is not this
project's to resolve.

An absent tag means two things at once, and both belong in the record: no
explicit grant was given, and **no explicit restriction was stated either**.

### What is and is not distributed

No model is distributed. There are **zero** model files in this repository - no
weights, no architecture, no checkpoint, in any format. The encoder was run
locally, once, and its outputs were kept.

What ships is `data/raw/auto_issue_seed/candidate_issue_profile.csv`: 88.1 KB,
247 rows, twenty numeric and categorical columns - candidate, issue, association
strength and evidence counts. No source sentence, no embedding, no weight.

### The problem that was anticipated

Two readings were foreseeable and neither is favourable if left unaddressed.

A reader could take "external-model-derived" to mean a model was redistributed.
It was not, and the section above says so in terms that can be checked against
the file list.

A reader could ask what basis the derived table ships on, and find it filed
under the project's own Apache-2.0 authorship - which would have been a false
claim, since Apache-2.0 describes this project's code and not an artefact
produced with someone else's encoder.

### How it was handled

The aggregate was moved out of `project_authored_and_derived_tables` into its
own source family, `external_model_derived_candidate_issue_aggregate`, whose
stated basis is what is actually relied on: that the file embeds no weight,
architecture or source sentence and so is not a copy or adaptation of the model.
No grant is claimed from the model author, and the missing tag is disclosed as
missing rather than read as permission.

`scripts/audit_public_data_rights.py` fails if this file is ever covered only by
a family whose basis does not mention a model. Adding that check immediately
caught that the file sat in two families at once.

### If the reading changes

Should the author later publish restrictive terms, or should a reviewer conclude
that a numeric aggregate falls inside them, the file is dropped and the model
runs without it. The consequence is measured rather than estimated: regional
macro MAE `2.613902987%p` to `4.935929128%p`, winner accuracy `0.8` to `0.6`.
Those figures are the V27-era measurement; `2.613902987` was V27's regional
macro, not V30's `2.566444753`. The removal has not been re-measured since, so
the magnitude is indicative rather than current.
That is why it is disclosed rather than quietly removed - the cost of removal is
real, and a reader is entitled to weigh it themselves.

### Proportion

The heavier rights question in this project is not the encoder. It is the
National Assembly proceedings the aggregate was computed over, whose provider
terms prohibit sharing supplied information unchanged. That is why the verbatim
corpus is excluded and only derived forms ship - see
`PRES_2025_INPUT_GUIDE.md`. Article 9 is satisfied independently of any of this:
the encoder is open-weight, it ran locally, and V30 executes no model at all.

## Submission checklist that remains outside code

- Submit both editable report (`HWP/HWPX` or `DOC/DOCX`) and PDF before the
  announced deadline.
- Keep the result report within five pages and attach the SBOM and, if judged
  applicable, the AI specification.
- Provide the public repository URL and a publicly viewable demonstration
  video URL.
- Confirm current-year duplicate public support truthfully; notify the
  organizer within the rule's reporting period if circumstances change.
- Keep team/member information consistent with the submitted application.

The repository cannot prove submission timing, video visibility, participant
identity or duplicate-benefit declarations. Those are explicit human checks.
