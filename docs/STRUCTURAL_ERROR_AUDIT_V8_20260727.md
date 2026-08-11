# Structural Error Audit for Active V8

Date: 2026-07-27

Active policy: `active_strict_nested_v8_universal_evidence_pipeline`

## Conclusion

Central regression remains a measurable structural error. It is not merely a
visual impression. Across the 199 candidate-region rows, the weighted slope of
predicted deviation from the equal-share baseline against actual deviation is
`0.8622`. A value of one would preserve the observed degree of polarization;
the active model preserves about 86% on average.

The sign pattern is also consistent with shrinkage:

- actual strong-side deviations of at least 10%p: mean error `-3.685%p`;
- actual weak-side deviations of at least 10%p: mean error `+1.800%p`.

The model therefore tends to underpredict strong candidates and regions while
overpredicting weak candidates and regions.

## Compression by election

| Election | Prediction-deviation slope | Interpretation |
|---|---:|---|
| 2002 | 0.880 | moderate compression |
| 2007 | 0.764 | strong compression |
| 2012 | 1.057 | slight over-dispersion, not central regression |
| 2017 | 0.769 | strong compression |
| 2022 | 0.929 | mild compression |

After removing each candidate's national mean and measuring only regional
shape, the overall slope is `0.8873`. The regional-shape slopes are `0.7416` in
2007 and `0.7355` in 2017. The error therefore exists both in national candidate
margin and in regional variation.

## National margin compression

| Election | Actual top-two margin | Predicted same-pair margin |
|---|---:|---:|
| 2002 | 2.434%p | -4.382%p |
| 2007 | 25.052%p | 11.200%p |
| 2012 | 3.547%p | 0.556%p |
| 2017 | 19.703%p | 11.921%p |
| 2022 | 0.759%p | 0.087%p |

V8 corrects four winners, but it still compresses the winning margin even in
the accurate 2012 and 2022 elections.

## Where the compression enters

Weighted deviation slopes for selected intermediate outputs are:

| Intermediate output | Slope |
|---|---:|
| `replacement_base_pred` | 0.9230 |
| `pre_hierarchy_pred` | 0.7097 |
| `shadow_pred` | 0.7501 |
| `camp_terrain_pred` | 0.7983 |
| pure `terrain_pred` | 1.0364 |
| blended `anchored_pred` | 0.8116 |
| final `layer_pred` | 0.8622 |

The prior terrain signal is not intrinsically too weak. Its unblended output
has nearly correct overall dispersion. Compression is introduced before the
terrain layer by hierarchy/postprocessing and then preserved because the
terrain anchor is blended at a low capped gain. Shock attenuation further
reduces the terrain anchor for 2017 even though a large shock can cause
consolidation as well as weaken historical attachment.

This means that globally sharpening the final softmax is not the appropriate
fix. The model should preserve reliable terrain selectively and adjust national
contest margins separately.

## Other remaining structural errors

### Third-candidate regional smoothing

The ratio of predicted to actual regional standard deviation is only `0.509`
for Lee Hoi-chang in 2007 and `0.611` for Ahn Cheol-soo in 2017. Third-candidate
national viability and regional concentration are still insufficiently
separated.

### Incorrect regional realignment

2012 is not mainly a central-regression failure. The model over-disperses
slightly and misplaces Chungcheong support. Chungnam has a paired error of
`14.36%p` and Chungbuk `9.36%p`. This points to candidate/regional realignment
mapping rather than insufficient extremity.

### National regime magnitude

The v8 regime layer corrects the 2007 winner but predicts only an `11.20%p`
Lee-Chung margin against the actual `25.05%p`. The 2017 Moon-Hong margin is
`11.92%p` against `19.70%p`. Governing-camp rejection is directionally correct
but its conversion into flexible-voter movement remains conservative.

### Early-era instability

2002 still predicts the wrong winner. The 1997 warmup and pre-2002 direct-party
history are too sparse and politically less comparable. This should not be
fixed by a 2002-specific gain.

### Predictor redundancy and discourse mismatch

In the 2007 early fold, `issue_advantage` and `rif` carry duplicate effective
information, while the party-context proxy rates Chung Dong-young above Lee
Myung-bak. The structural terrain and regime layers compensate for this, but
the underlying candidate/discourse layer remains miscalibrated.

## Defensible next experiment

1. Separate regional-shape calibration from national candidate-margin
   calibration.
2. Replace the fixed terrain-anchor cap and scalar shock divisor with a dated
   reliability rule based on the number, recency, and type of prior direct-party
   elections.
3. Learn or validate that rule using pre-election proportional and local-council
   party ballots, not presidential target outcomes.
4. Preserve a conservative core floor while allowing critical and swing layers
   to expand when prior-party erosion and explicit regime-rejection evidence
   agree.
5. Add a separate regional-concentration model for third candidates.
6. Evaluate 2012 Chungcheong realignment as a direction error, not by increasing
   every regional multiplier.

No active code, configuration, or prediction was changed during this audit.
