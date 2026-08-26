<!-- active-model-version: v32 -->
# Workspace Structure

| 경로 | 현재 역할 |
|---|---|
| `src/election_forecast/` | 설치 가능한 공개 패키지, V31 CLI와 검증된 런타임 로더 |
| `presidential_issue_engine/` | 대통령선거 특징 조립, PIT 처리, 지역·후처리 계층 |
| `presidential_issue_engine/forecast_time_region_weights.py` | 종단 변환이 쓰는 예측 시점 지역 가중(직전 선거 투표량) |
| `scripts/run_current_presidential_model.py` | 현재 V31의 소스 체크아웃 실행 진입점 |
| `data/config/current_presidential_model.json` | 유일한 정식 활성 버전 포인터 |
| `data/config/active_presidential_model.json` | 위 정식 포인터의 호환 별칭 |
| `data/config/active_presidential_model_v16.json` | 후대 버전 실행 계보가 사용하는 동결 V16 내부 기반 설정 |
| `outputs/active_presidential_nested_v31/` | 변경 금지 V31 산출물과 승격 감사 |
| `outputs/active_presidential_nested_v23/` ~ `v30/` | 변경 금지 롤백 증거 |
| `outputs/prospective_pres_2025_v31/` | 결과 확인 뒤 수정된 demonstration; prospective 검증이 아님 |
| `research/` | 설치 배포물에서 제외되는 비승격 연구와 구식 시각화 |
| `docs/` | 설계, 데이터 권리, 규정 준수, 재현·승격 기록 |
| `tests/` | 회귀, PIT, 누수, 패키지·공개 경계 검사 |

공개 저장소와 설치 배포물의 정확한 포함 범위는
`docs/REPOSITORY_BOUNDARIES.md`를 따른다. 로컬 캐시, 자격증명, 원문 말뭉치,
권리가 불명확한 행 단위 자료는 어느 공개 배포물에도 포함하지 않는다.
