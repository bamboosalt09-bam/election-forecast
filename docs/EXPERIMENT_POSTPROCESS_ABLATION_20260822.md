# Postprocess ablation and order permutation

## Status

- Date: 2026-08-22
- Status: diagnostic; **no change made to the model**. V26 keeps the shipped
  order `veto > ceiling > refusal`.
- Post-2022 outcomes used: none
- Reproduce with `python scripts/evaluate_postprocess_ablation.py`

The three structural postprocesses are not commutative, so isolated on/off
ablation cannot describe them. This measures both dimensions at once: every
on/off subset, and every ordering of the subsets larger than one. Sixteen cells.

The nested run happens once. The three layers are pure transforms applied after
it, so every cell reuses one captured pre-postprocess frame; the frame is taken
by intercepting the first layer rather than reconstructed, so it is exactly what
V26 feeds the stack.

## The layers barely co-occur

This is the finding everything else follows from.

| election | veto | ceiling | refusal | layers firing |
| --- | :-: | :-: | :-: | :-: |
| pres_2002 | · | O | O | 2 |
| pres_2007 | O | · | · | 1 |
| pres_2017 | O | · | · | 1 |
| pres_2022 | · | · | O | 1 |
| **pres_2025** | **O** | **O** | **O** | **3** |

**No scored election fires all three.** Exactly one fires two. The forecast
target fires all three.

## Results

| cell | regional weighted macro | national macro | winners |
| --- | ---: | ---: | ---: |
| refusal>veto>ceiling | 2.676769 | **0.709042** | 4/5 |
| veto>refusal>ceiling | 2.676769 | **0.709042** | 4/5 |
| refusal>ceiling>veto | 2.676769 | **0.709042** | 4/5 |
| **veto>ceiling>refusal** (shipped) | 2.712233 | 0.720994 | 4/5 |
| ceiling>veto>refusal | 2.712233 | 0.720994 | 4/5 |
| ceiling>refusal>veto | 2.712233 | 0.720994 | 4/5 |
| veto>refusal / refusal>veto | 2.771157 | 0.771775 | 4/5 |
| veto>ceiling / ceiling>veto | 3.156698 | 1.309632 | 4/5 |
| refusal>ceiling | 2.913759 | 1.329852 | 4/5 |
| ceiling>refusal | 2.949224 | 1.341803 | 4/5 |
| refusal | 3.008147 | 1.392584 | 4/5 |
| veto | 3.715606 | 1.858003 | 4/5 |
| ceiling | 3.393689 | 1.930441 | 4/5 |
| none | 3.952597 | 2.478813 | 4/5 |

### The veto's apparent commutativity is an artifact

`veto>ceiling` and `ceiling>veto` agree exactly. So do `veto>refusal` and
`refusal>veto`. That is not a property of the transforms - it is because the
veto never co-fires with another layer on the scored panel. The only genuinely
non-commutative pair is ceiling and refusal, which is also the only pair that
shares an election.

The six full orderings collapse into exactly two values, and the split is
decided by one thing: whether refusal runs before ceiling (0.709042) or after
(0.720994). The veto's position is irrelevant everywhere.

### The whole ordering effect is one election

| election | shipped | best ordering | difference |
| --- | ---: | ---: | ---: |
| pres_2002 | 2.3416 | 2.2818 | 0.0598 |
| pres_2007 | 0.6610 | 0.6610 | 0.0000 |
| pres_2012 | 0.1271 | 0.1271 | 0.0000 |
| pres_2017 | 0.2011 | 0.2011 | 0.0000 |
| pres_2022 | 0.2741 | 0.2741 | 0.0000 |

2002 is the only scored election where two layers meet, so it is the only place
an ordering can matter. **The shipped order was not changed.** Reordering to
gain 0.0120 %p of national macro, all of it from 0.0598 %p on a single
election, is the noise-fitting this harness exists to expose.

### Winner accuracy is insensitive to the entire stack

All sixteen cells score 4/5, including `none`. The published winner accuracy is
not produced by any of this machinery, and no ordering or subset of it can move
that number on this panel.

### Isolated contributions, against `none` at 2.478813

| layer | national macro | improvement |
| --- | ---: | ---: |
| refusal | 1.392584 | **1.086229** |
| veto | 1.858003 | 0.620810 |
| ceiling | 1.930441 | 0.548372 |

Refusal is the largest single contributor by roughly a factor of two. The veto
is the most visible layer and the second-largest. The full stack at 0.720994
beats every single layer, so on this panel the three are complementary rather
than redundant.

## What this does not settle

The redundancy concern is about 2025, where the veto widens the two-major gap
from 7.80 to 17.48 points and the ceiling then redistributes the third
candidate's recovered mass at that already-widened ratio, adding a further
3.18. **That is a three-layer configuration with zero scored observations.** The
panel cannot test it, and this harness makes the reason explicit rather than
resolving it.

Two consequences worth stating plainly:

1. Any change justified by the 2025 interaction is justified by an
   unobservable. It can be argued from mechanism, but not from the panel.
2. The per-layer activation counts - veto 2, ceiling 1, refusal 2 - are the
   honest measure of how much evidence each structural rule rests on. They
   belong beside the metrics wherever the structural layers are described.
