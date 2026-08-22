# District Terrain and Candidate Regionalism

> Historical pre-electorate-layer experiment. Current metrics and policy are in
> `docs/FINAL_MODEL_V27_20260822.md`.

## Purpose

지역 정당 지형, 지역구 선거의 후보 중심 지형, 대선 후보 개인의 지역 기반을 한
변수로 섞지 않는다. 서로 다른 투표 행위를 별도 층으로 계산한 뒤, 사전정보만으로
후보별 지역 구성 차이를 설명한다.

## Three Regional Layers

1. Party terrain prior

   비례대표·전국구·지방의회 비례처럼 정당에 직접 투표한 결과를 중심으로 정당의
   지역 지지 기반을 만든다. 대통령 후보의 개인 효과가 아니라 정당 기본 지형이다.

2. District-election terrain

   총선 지역구, 광역·기초의회 지역구, 광역·기초단체장처럼 후보에게 투표한 선거만
   별도로 집계한다. 사용 가중치는 총선 지역구 `1.00`, 광역의회 지역구 `0.80`,
   기초의회 지역구 `0.50`, 광역단체장 `0.15`, 기초단체장 `0.10`이다. 비례대표
   결과는 이 층에서 제외한다.

3. Candidate personal regionalism

   사전 국회 발언·정치 지형과 후보별 지역 기반을 사용한다. A/B 후보는 개인 지역
   기반을 그대로 유지한다. C 후보는 전국 경쟁력 gate와 같은 진영 분열 정도를
   함께 보고 `max(gate, sqrt(bloc_split))`를 적용한다. 따라서 전국 경쟁력이 낮다는
   이유만으로 2007년 이회창 같은 진영 분열형 후보의 충청 지역 기반이 소거되지
   않는다.

## Leakage and Normalization Guards

- 모든 선거 이력과 후보 정치 지형은 목표 대선 D-1 이전 자료만 사용한다.
- district terrain은 선거×후보와 선거×지역 양쪽으로 중심화한다.
- candidate regionalism도 후보×지역 구성 신호로 정규화한다.
- 두 신호 모두 후보의 전국 득표율을 직접 올리는 보너스로 사용할 수 없다.
- 2025 이후 결과는 선택, 비교, ablation에 사용하지 않는다.

## Selection Result

직접 district-terrain scale 후보 `0`, `.05`, `.10`을 nested rolling 안에서
검토했으나 일반화 개선 기준을 통과하지 못했다. 따라서 기능과 진단열은 유지하되
최종 배포 scale은 `0`이다. 지역구 자료를 넣었다는 이유만으로 강제 활성화하지
않는다.

채택된 개선은 후보 개인 지역 기반의 C후보 factor 수정이며, 기존 선택값
`regionalism_scale=0.15`, `anchor=1.0` 안에서 동작한다.

## Intermediate Verified Change

아래 표는 within-bloc regional transfer를 추가하기 전, 지역주의 층만 수정한 중간
스냅샷이다. 당시 후속 성능은 `WITHIN_BLOC_REGIONAL_TRANSFER.md`를 따랐다.

| Metric | Before | After |
|---|---:|---:|
| Nested rolling row MAE | 4.493%p | 4.491%p |
| Selection-sample rolling row MAE | 4.051%p | 4.011%p |
| LOEO row MAE | 5.017%p | 4.963%p |
| 2007 rolling row MAE | 5.623%p | 5.457%p |

2007 C후보의 충남 오차는 `-17.28%p`에서 `-15.25%p`, 대전은
`-13.70%p`에서 `-11.66%p`로 줄었다. 부산·경남 등에는 여전히 큰 잔여오차가
있으므로 지역주의 문제를 해결 완료로 표현하지 않는다.

당시 검증 스냅샷은 strict PIT `215/215`, 전체 테스트 `246 passed`였다.
