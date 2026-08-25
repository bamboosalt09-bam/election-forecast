<!-- active-model-version: v31 -->
# GitHub Workflow

## Repository role

This repository is the source of truth for code, tests, model configuration,
small canonical inputs, and the frozen artifacts needed to audit active V31.
V27 through V23 remain tracked as immutable rollback boundaries.
OneDrive and local backup directories are recovery storage only and are not Git
working trees.

## Branches

- `main` contains only reviewed, reproducible states.
- Work is performed on `codex/<topic>` or another short-lived topic branch.
- Changes reach `main` through a pull request after required CI checks pass.
- Force-pushes and direct model edits on `main` are not part of the workflow.

## Model lifecycle

V31 is frozen and V30 through V23 remain frozen rollback models. Experiments
must use a new versioned output directory and must not overwrite any frozen
version's code, configuration, predictions, or manifests. Promotion of a later version requires
all of the following in one pull request:

1. a new versioned configuration and runner, or an explicitly declared frozen
   base configuration with a versioned wrapper;
2. strict chronological nested evaluation;
3. point-in-time and target-outcome invariance audits;
4. a performance report using the active metric definitions;
5. an updated current-model pointer and a promotion manifest;
6. an immutable Git tag after merge.

The 2025 presidential outcome is prohibited from fitting, tuning, ablation,
model selection, and pre-evaluation comparison.

## Data boundary

Git tracks small canonical inputs, the active V31 audit bundle, and the frozen
V27 through V23 rollback bundles. It does not
track full transcript corpora, API caches, virtual environments, shadow
classifier corpora, backups, or bulk experiment outputs. Those files remain in
external storage and are identified by source metadata and checksums when they
are required for a reproducible experiment.

## CI tiers

The normal GitHub-hosted CI runs the repository-boundary audit, the regression
suite, and the frozen V31 audit. Full Assembly reprocessing and historical NLP
experiments are manual jobs for a controlled local or self-hosted runner. A
self-hosted runner must not execute untrusted pull-request code.

## Pull-request record

Every model-affecting pull request records the hypothesis, changed inputs,
available-date boundary, tests, strict nested metrics, rejected alternatives,
and whether the active pointer changed. A result is not promoted merely because
one historical election improves.
