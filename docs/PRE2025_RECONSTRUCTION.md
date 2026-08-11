# Initial Clean Reconstruction Record

이 문서는 현재 재도출 작업보다 앞선 1차 clean 분리 기록이다.

1. 분리 전 전체 작업공간은
   `C:\english_folder\poll_project_post2025_outcome_aware_20260714`로 보존했다.
2. 실제 결과 행과 이후 관측 행을 제거하고 삭제 전후 해시를 기록했다.
3. 중립화한 상태는
   `C:\english_folder\poll_project_through2022_baseline_locked_20260714`에 다시 잠갔다.
4. 활성 엔진에서는 기존 특징 구조의 레이어 강도를 2002~2022만으로 재도출했다.

감사 파일:

- `PRE2025_SANITIZATION_AUDIT.json`: 제외 행의 삭제 전후 해시
- `PRE2025_ASOF_AUDIT.json`: 기준일 이후 관측치 삭제 전후 해시
- `THROUGH2022_REDERIVED_MANIFEST.json`: 현재 활성 엔진 파일 해시

현재 모델 해석과 성능은 `THROUGH2022_REDERIVATION.md`를 우선한다.
