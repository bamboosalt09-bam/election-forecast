# Through-2022 Layer Rederivation

> Historical pre-electorate-layer selection record. Current metrics are in
> `docs/CURRENT_MODEL_PERFORMANCE_20260716.md`.

## What Was Preserved

분리 전 엔진의 Ridge 변수, 특징 계산식, 지역 prior, 이슈·거시·세대·후보 문맥,
제3후보 및 지역주의 특징 구조는 그대로 유지했다. 대용량 회의록을 다시 처리하지
않고 이미 생성된 PIT-safe aggregate를 사용했다.

## What Was Rederived

이후 결과를 보고 활성 강도가 선택됐을 가능성이 있는 아홉 설정을 중립값에서 다시
선택했다.

| Layer | Grid | Selected |
|---|---|---|
| Ridge alpha | .05, .10, .20, .30, .50, .80, 1.20 | 1.20 |
| Sparse region residual | off, scale .5/1.0, shrinkage 4/8/16 | off |
| Neutral issue context | scale 0, .20, .40, .60, .80 | .60 |
| Issue-character overlay | gain 0, .08, .12, .16, .24, .32 | 0 |
| Vote conversion | scale 0, .01, .02, .035, .05 | 0 |
| District-election terrain | scale 0, .05, .10 | 0 |
| Candidate regionalism | scale 0~.15, anchor 1, 2, 3.5 | .15 / 1.0 |
| Third-candidate gate / character | off/off, on/off, off/on, on/on | on/on |
| Within-bloc regional transfer | scale 0, .20, .35, .50; stronghold 0~.50 | .50 / .25 |
| Issue seed source | manual or automatic | automatic |

선택 순서와 grid는 outer 결과를 계산하기 전에 고정했다.

## Nested Results

| Outer target | Tuning elections | Selected additions | Row MAE |
|---|---|---|---:|
| 2002 | none | neutral | 4.171%p |
| 2007 | 2002 | neutral | 5.598%p |
| 2012 | 2002, 2007 | alpha .8 + regionalism + transfer .50 / stronghold 0 | 5.190%p |
| 2017 | 2002~2012 | alpha 1.2 + regionalism + transfer .50 / stronghold .10 | 4.746%p |
| 2022 | 2002~2017 | alpha 1.2 + neutral context + regionalism + third gate/character + transfer .50 / stronghold .25 | 2.145%p |

Row-count weighted nested MAE는 `4.491%p`다. 모든 재선택 대상 설정을 중립화한
baseline의 동일 범위 rolling MAE는 `5.686%p`다. 두 값은 평가 구조가 다르므로
단순한 holdout 개선으로 해석하지 않는다.

전체 허용 표본으로 선택한 최종 설정의 rolling MAE는 `3.710%p`, LOEO는
`4.980%p`다. 이 rolling 값은 selection-sample 수치이며 외부 holdout이 아니다.

지역구 선거만으로 만든 직접 지형층은 구현했지만 nested 선택에서 개선 기준을
통과하지 못해 최종 scale은 `0`이다. 후보 개인 지역 기반은 전국 제3후보 경쟁력과
분리해, 같은 진영 분열형 후보의 지역 조직이 전국 경쟁력 gate만으로 사라지지 않게
수정했다. 상세 식과 검증은 `DISTRICT_TERRAIN_AND_CANDIDATE_REGIONALISM.md`에 있다.

동일 진영 지역표 재배분은 2002의 약한 분산 신호와 2007의 강한 진영 분열 신호를
연속적으로 사용한다. 최소 두 선거에서 activation이 `0.001` 이상이어야 선택하며,
최대 scale은 `0.50`으로 제한했다. 전국 제3후보 gate를 먼저 정한 뒤 지역 transfer를
선택하고, 문서화된 후보 개인 기반은 제곱형 stronghold 항으로 증폭한다. 최종
stronghold gain은 `0.25`다. 이 층은 2007 selection-sample 오차를 크게 낮추지만
이후 활성 선거가 없어 nested MAE 자체는 바꾸지 않는다. 따라서 개선된 `3.710%p`를
외부 검증값으로 해석하지 않는다.

## Remaining Limitation

데이터와 metric 경계는 코드로 차단했지만, 특징 구조의 아이디어 자체가 이후 결과를
본 뒤 만들어졌다는 사실까지 되돌릴 수는 없다. 따라서 이 모델은
`outcome-blind architecture`가 아니라 `through-2022 rederived weights on a fixed
architecture`로 표현해야 한다.

## Artifacts

- final config: `data/config/through2022_rederived_layers.json`
- nested folds: `presidential_issue_engine/report/through2022_rederived/nested_outer_results.csv`
- full search trace: `presidential_issue_engine/report/through2022_rederived/selection_trace.csv`
- machine-readable summary: `presidential_issue_engine/report/through2022_rederived/summary.json`
- workspace hashes: `docs/THROUGH2022_REDERIVED_MANIFEST.json`
