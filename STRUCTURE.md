# Workspace Structure

| 경로 | 역할 |
|---|---|
| `src/` | 공개 forecast 패키지와 데이터 수집·특징 코드 |
| `presidential_issue_engine/` | 대통령선거 예측, rolling/LOEO, PIT 감사 |
| `data/` | 활성 입력 데이터와 설정 |
| `common/` | 공통 지역·스키마 유틸리티 |
| `scripts/` | 데이터 생성, 감사, 실험 보조 스크립트 |
| `tests/` | 단위·통합·누수 방지 테스트 |
| `docs/` | baseline 출처, 삭제 감사, 해시 매니페스트 |

대용량 산출물, 과거 보고서, 통계대회 제출본, 기존 Git 이력은 활성 폴더에 복사하지
않았다. 그것들은 보존 폴더
`C:\english_folder\poll_project_post2025_outcome_aware_20260714`에 그대로 남아 있다.
