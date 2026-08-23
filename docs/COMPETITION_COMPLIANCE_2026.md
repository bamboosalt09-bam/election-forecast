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
| Article 8: disclose all third-party software, sources and licenses | Human-readable direct-dependency SBOM plus exact tested lock | `docs/SBOM.md`, `requirements-v27.lock` |
| Article 9: embedded AI must run independently and be at least open weights | Active V28 uses no external AI weight, external-model-derived overlay or hosted inference API; historical experiments are excluded and disclosed | `docs/AI_MODEL_SPEC.md` |
| Article 10: complete source must be public and reviewable | Public source tree plus a self-contained wheel that embeds the actual V28 runtime | `setup.py`, `src/election_forecast/v28_runtime.py` |
| Article 10: winning repository remains public for five years | Repository policy records the obligation; deletion/private conversion must not occur during that period | this document and release checklist |
| Article 11: repository state at evaluation time is judged | `main` is protected; mandatory checks pin current-pointer consistency, audit, reproduction, package and rights boundaries | `.github/workflows/ci.yml`, `scripts/audit_current_public_surface.py` |
| Result-report form: architecture, stack, data flow, run/test and limitations | Current V28 documents link actual entry points and evidence boundaries | `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md` |
| Attachment 1: SBOM is mandatory | Direct dependency table and exact lock are maintained | `docs/SBOM.md` |
| Attachment 2: AI model specification when applicable | V28 statistical model and non-active external-model experiments are distinguished | `docs/AI_MODEL_SPEC.md` |
| Duplicate-benefit confirmation | Participant must disclose same/substantially similar current-year public support, including pending/ongoing participation | submission-side declaration; not inferred by code |

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
