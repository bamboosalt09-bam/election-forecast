# Direct Mega-Issue Shift

## Root cause

The 2017 intensity input was already `2.0`, but it did not produce the intended
large effect:

1. Automatic mega-axis weights were between `0.55` and `1.0`, while the engine
   boosted only the amount above `1.0`. The salience boost was therefore zero.
2. The active runner regenerated the signed candidate profile but did not turn
   on the full enhanced-issues branch.
3. Turning on that full branch was not safe. Ridge could learn a negative
   coefficient for a feature whose sign already represented candidate benefit
   or burden, reversing its semantic direction.

## Implemented rule

The active model now compiles one direct mega score per eligible election:

```text
score = direction * association_strength * confidence * mega_issue_intensity
```

Eligibility requires all of the following:

- explicit person, party, or government target evidence;
- a political-shock issue type;
- `available_date <= election day - 1`;
- pre-election `mega_issue_intensity > 1.0`.

When multiple issues qualify, the issue with the strongest
`target_attribution_confidence * log1p(target_absolute_evidence)` is selected.
The score is capped at `+/-0.50`. The final candidate share is adjusted in log
space with gain `0.40`, capped at `+/-0.20`, and every regional contest is
renormalized to 100%.

## Current activation

Only one through-2022 row passes the gate:

| Election | Slot | Issue | Intensity | Score | Log shift |
|---|---|---|---:|---:|---:|
| 2017 | B | regime_change | 2.0 | -0.254832 | -0.101933 |

The shift lowers the incumbent-conservative continuity candidate burdened by
the explicitly attributed regime-change issue. The other four scored elections
receive exactly zero direct shift.

## Strict nested result

| Metric | Before | After |
|---|---:|---:|
| 2017 regional vote-weighted MAE | 7.919%p | 6.789%p |
| 2017 national point MAE | 7.069%p | 5.863%p |
| Regional equal-election macro MAE | 5.808%p | 5.582%p |
| National equal-election macro MAE | 4.810%p | 4.569%p |
| Winner accuracy | 20% | 40% |

2002, 2007, 2012, and 2022 are unchanged.

## Rejected alternatives

- Boosting every top-salience axis worsened national macro MAE to 4.870%p.
- Enabling the full signed enhanced-issues branch worsened national macro MAE
  to 5.717%p and 2017 national MAE to 14.597%p.
- Applying direct shifts to intensity-1 elections improved 2017 but damaged
  2002 and 2022.

## Interpretation limit

This is a through-2022 development change. Only 2017 currently exceeds the
intensity gate, so the gain does not have multiple independent high-intensity
elections for validation. It is a bounded structural correction supported by
point-in-time inputs, not an untouched holdout result.
