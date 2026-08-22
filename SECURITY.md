# Security policy

## Supported release

Security fixes are applied to the current `main` branch and the active package
release. Frozen V23-V27 research artifacts are immutable evidence: a security
fix that changes model behavior must be released as a new version rather than
rewriting a frozen artifact.

## Reporting a vulnerability

Do not open a public issue for credentials, arbitrary code execution, unsafe
archive handling, dependency compromise, or disclosure of non-public source
material. Use GitHub's private vulnerability-reporting form for this
repository. Include the affected command, version, operating system, minimal
reproduction and expected impact. Please do not include real personal data or
credentials in the report.

## Security boundary

- The public V27 runner is local and does not require a hosted inference API.
- The packaged runtime admits Git-tracked files only, rejects path traversal,
  and verifies every extracted file with SHA-256 before execution.
- API credentials belong in local environment variables. `.env`, caches,
  checkpoints, full-text parliamentary corpora and raw market exports are not
  distributed.
- Optional historical stance-model experiments are not part of the active V27
  runtime or the competition demonstration.

Automated checks scan the public boundary, source/data licenses, pinned GitHub
Actions, dependency updates, the frozen V27 artifact and clean reproduction.
