# Within-bloc Regional Transfer

> Historical pre-electorate-layer experiment. Current metrics are in
> `docs/CURRENT_MODEL_PERFORMANCE_20260716.md`.

## Purpose

지역 정당 지형이 계산된 뒤, 같은 성향 후보가 둘 이상 출마한 경우 진영 지역표를
후보 사이에서 재배분한다. 지역 prior 총량을 다시 키우지 않고 후보 귀속만 바꾼다.

## Activation

제3후보 C의 사전 성격값으로 다음 activation을 계산한다.

`activation = competitiveness * max(bloc_split - independent_pole, 0)`

| Election | Activation | Interpretation |
|---|---:|---|
| 2002 | 0.0053 | 약한 진보·개혁 표 분산 |
| 2007 | 0.1099 | 강한 보수 진영 분열 |
| 2012 | 0 | 양강 구도 |
| 2017 | 0 | 독립된 제3축이 bloc split보다 우세 |
| 2022 | 0 | 단일화 후 양강 구도 |

2002와 2007을 같은 강도로 처리하지 않는다. activation이 `0.001` 이상인 과거
선거가 두 개 이상일 때만 scale 선택을 허용한다.

## Regional Profile

C후보의 후보 개인 지역주의 신호와 지역구 선거 지형을 각각 후보 내에서 중심화·
정규화한 뒤 동일 비중으로 결합한다. 지역별 C 증가분은 정치 성향 affinity가 있는
A/B 후보에게서만 차감한다. 각 지역에서 활성 후보 예측값은 다시 합계 100%로
정규화된다.

이 기본 profile에 두 사전정보 확인항을 더한다.

- same-lane reservoir: 같은 성향 주류 후보의 선거 전 `partisan_prior`와 과거 선거
  유효 개수로 진영 저수지를 구한다. 기본 profile과 방향이 같은 지역만 보강하고,
  강한 개인 기반 지역은 중심화 차감에서 보호한다.
- personal stronghold: `candidate_regional_base_gated`를 후보 내 최대값으로 나눈 뒤
  제곱한다. 이미 강한 개인 기반만 비선형적으로 증폭하고, 총량 상쇄는 비강세
  지역에서만 수행한다.

`profile = base + 1.00 * reservoir_confirmation + 0.25 * stronghold_reinforcement`

전국 제3후보 경쟁력과 성격을 먼저 선택한 뒤 지역 transfer를 선택한다. 최종 scale
grid는 `0`, `.20`, `.35`, `.50`, stronghold gain grid는 `0`~`.50`이며 채택값은
각각 `.50`, `.25`다. 이 순서는 전국 경쟁력을 지역기반으로 다시 정의하지 않고,
정해진 후보 총량의 지역 귀속만 조정한다.

## Results

| Metric | Transfer 전 | 기본 transfer | 현재 |
|---|---:|---:|---:|
| Nested rolling row MAE | 4.491%p | 4.491%p | 4.491%p |
| Selection-sample rolling row MAE | 4.011%p | 3.764%p | 3.710%p |
| LOEO row MAE | 4.963%p | 4.950%p | 4.980%p |
| 2002 rolling row MAE | 4.041%p | 4.073%p | 4.065%p |
| 2007 rolling row MAE | 5.457%p | 4.409%p | 4.190%p |
| 2007 C-slot row MAE | 7.368%p | 5.326%p | 4.914%p |

2012, 2017, 2022는 activation이 0이므로 결과가 바뀌지 않는다. 현재 2007 C후보
오차는 충남 `-9.65%p`, 대전 `-6.00%p`, 충북 `-2.04%p`, 대구 `-6.93%p`다.
stronghold gain을 0에서 `.25`로 올린 효과만 보면 대전·충북·충남의 C 예측이 각각
`1.374%p` 증가했고 세 지역의 절대오차가 같은 폭으로 감소했다. 호남 C 과대평가는
약 `+1.2~1.5%p`까지 줄었다.

## Limits

- 이후 활성되는 진영 분열선거가 없어 nested outer MAE는 개선되지 않는다.
- selection rolling 개선의 대부분은 2007에서 발생하므로 외부 검증으로 부르지 않는다.
- 부산·경남의 C 과소평가는 여전히 약 `8.8~9.8%p` 남아 있다.
- 호남에서 C가 줄어든 표를 같은 성향 A에게만 반환하므로 B 과소평가는 남는다.

## Consistency Guard

엔진과 `robustness_check.py`가 같은 `apply_prediction_postprocess`를 사용하도록
통합했다. 저장된 rolling CSV의 MAE와 엔진 metric은 회귀 테스트에서 `1e-10`
이내로 일치해야 한다.

현재 검증은 strict PIT `215/215`, 전체 테스트 `251 passed`다.
