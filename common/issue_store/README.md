# issue_store — 이슈 메모리 (공유)

"이슈를 모아 메모리에 저장하고 분석한다"의 그 메모리. 날짜가 붙은 이슈/사건 기록이
누적되고, 시간감쇠 + 지역 민감도로 굴려 예측 변수로 바뀐다.

## 핵심: 스토어는 하나, 채우는 방법(populator)은 셋

```
 ┌─ curated   (사람이 핵심 이슈를 직접 입력)        ┐
 ├─ aggregate (BIGKinds 이슈분석 + 검색 트렌드)     ├─►  IssueEventRow  ─► rollup ─► feature_schema ─► engine
 └─ corpus    (수십만 본문 분석 → news_analyzer)    ┘     (같은 스키마)      (같은 수식)
```

세 populator가 **같은 `IssueEventRow`** 를 쓰고, **같은 `rollup_issue_features`** 가
변수로 만든다. 그래서 단계적 전략이 성립한다:

| 단계 | populator | 대회 | compute |
| --- | --- | --- | --- |
| 지금 | `curated` / `aggregate` | open-source forecast | 거의 0 |
| 나중 | `corpus` | 오픈소스대회 | 말뭉치 + GPU 예산 |

## 한 줄(IssueEventRow)이 담는 것

- 무엇: `issue_name`, `issue_type`(policy/scandal/endorsement/unification/...)
- 언제: `event_date`, `available_date`(누수 차단 — 예측 시점 이후 이슈는 rollup에서 제외)
- 누구: `slot`(A/B/C/alpha), 선택적 `candidate_id`
- 어디: `region_scope`("ALL" 또는 특정 region_id)
- 얼마나: `salience_score`(노출량), `direction_score`(호오 −1..1),
  `candidate_link_score`(연결도), `media_reliability_score`(매체 신뢰도)
- 출처: `populator`, `confidence`, `source_note`

점수 컬럼명은 엔진의 기존 `issue_scores.csv` 와 일치시켜, `election_forecast`
가 변경 없이 소비한다.

## rollup 수식 (기존 엔진과 동일)

1. `available_date <= forecast_date` 만 사용 (누수 차단).
2. `final_issue_score = salience × direction × media_reliability` (없으면 유도).
3. 시간감쇠 `weight = exp(-age / half_life)`.
4. `(election_id, slot, issue_name)` 가중 평균 → `region_issue_sensitivity` 결합.
5. `component = issue_score × sensitivity × candidate_link`.
6. 라우팅: 위험형(scandal/gaffe) → `risk_or_negative`(음의 부담), 그 외 →
   `local_issue_fit`. `tanh` 로 [-1, 1] 보장.

## 사용 예

```python
from common.issue_store import validate_issue_frame, rollup_issue_features

validate_issue_frame(issues)                      # populator 출력 검증
features = rollup_issue_features(issues, region_sensitivity, forecast_date="2022-03-08")
# features 는 그대로 feature_schema 모양 → 엔진/평가로 직행
```
