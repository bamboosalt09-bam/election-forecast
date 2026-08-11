"""Shared contracts for the Jeongukgu election-forecasting projects.

`common` is intentionally thin. It holds ONLY what both competitions must agree
on so their results stay comparable:

- `shared_schema`      : election-type / contest / aggregation-rule dimensions.
- `election_slot_schema`: the A/B/C/alpha analytic slot definition.
- `feature_schema`     : the "numericalization contract" every scorer must emit.
- `issue_store`        : the populator-agnostic "issue memory" + rollup to features.
- `evaluation`         : the single scoring ruler (MAE, %p error, winner accuracy).

What is deliberately NOT here: the numericalization layer (rule-based vs AI) and
the forecast engine itself. Those diverge by design between the statistics
competition (simple, presidential-only, AI-free) and the open-source competition
(more complex, multi-election, open-weight AI allowed).

See `common/README.md` for the boundary rationale.
"""

__all__ = [
    "shared_schema",
    "election_slot_schema",
    "feature_schema",
    "issue_store",
    "evaluation",
]
