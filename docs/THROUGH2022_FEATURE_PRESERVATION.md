# Through-2022 Feature Preservation

> Historical pre-electorate-layer snapshot. Current metrics are in
> `docs/CURRENT_MODEL_PERFORMANCE_20260716.md`.

## Principle

활성 엔진은 분리 전까지 개발한 모델 구조와 입력 특징을 유지한다. 삭제 대상은
2025 대선 결과 행, 그 결과를 읽는 평가 코드, 2025 결과를 보고 선택한 가중치다.
기존 기능을 단순 모델로 되돌리거나 과거 실험을 처음부터 폐기하지 않는다.

## Preserved Model Work

- 대선·총선 비례/전국구·지역구·지방의회·단체장 결과를 당 지형 prior로 집계하는 구조
- 지역별 콘크리트 지지층과 일반 지지층, 정당 지형과 후보 개인 지역주의의 분리
- 경제·무역·금리·주택·KOSPI 지표와 시대별 중요도 및 이슈 보강 신호
- 15대 이후 국회 발언 자료, 발언자 선수·직위·지역구/비례·험지/텃밭 이력 가중
- 거대 이슈 유형·충격량·긍정/부정·시간 감쇠·잔류 효과
- 자당 의원의 후보 대우, 타당의 후보 대우, 외부 인식, 당내 결집과 분산 위험
- 후보 정치 지형, 세대 지형, 화제성 대비 득표 전환, 제3후보 성격과 단일화 처리
- 문장 분류기, 연속 stance 강도, 중립 문맥 및 이슈 성격 산출물
- 지역별 합계 100% 정규화와 계수·잔차 불확실성을 포함한 Monte Carlo 구간

위 기능의 코드와 2022년까지의 PIT-safe 산출물은 활성 폴더에 남아 있다. 긴 회의록
재처리는 하지 않았으며 기존 생성물을 사용한다.

## Rederived Weights

2025 관찰 이후 영향을 받았을 가능성이 있는 다음 설정만 1997 warmup과
2002~2022 scored 선거로 다시 선택했다.

- Ridge alpha
- sparse region residual calibration
- neutral issue-context scale
- issue-character overlay gain
- candidate vote-conversion scale
- district-election terrain scale
- candidate regionalism scale and anchor
- within-bloc regional transfer scale
- within-bloc same-lane reservoir and personal-stronghold gain
- third-candidate gate/character mode

당내 문맥·외부 대우·세대 등 2025 검증 이전에 구축되어 기존 과정의 일부였던 층을
모두 0으로 초기화하지 않는다. 이를 전부 중립화한 진단은 초기 rolling 폴드의 기존
정보까지 제거하므로 배포 설정으로 사용하지 않는다.

## Source Comparison

분리 전 보존본과 활성 폴더의 Python/설정 소스를 해시 대조한 결과는 다음과 같다.

- identical files: 225
- active-only boundary/audit files: 6
- changed for PIT filtering or through-2022 configuration: 24
- archive-only files: 6

archive-only 파일은 2025 holdout 실행, 2025 지역 정확도 평가, 2025 stance 작성 및
2025 결과를 포함한 제3후보 진단이다. 이 파일들은 모델 핵심 구조가 아니라 금지된
결과를 읽는 평가 도구이므로 활성 폴더에 복원하지 않는다.

## Verification

```powershell
python scripts\rederive_layers_through2022.py
python presidential_issue_engine\audit_weight_selection_boundary.py
python presidential_issue_engine\audit_point_in_time.py --deep
python presidential_issue_engine\robustness_check.py
pytest -q
```

현재 검증값은 nested rolling `4.491%p`, selection-sample rolling `3.710%p`,
LOEO `4.980%p`, strict outcome-invariance `215/215`다. 이슈 seed는 수동 입력 대신
국회 발언 집계 기반 자동 생성물을 사용한다.
