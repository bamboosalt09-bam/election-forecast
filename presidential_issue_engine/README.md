# Presidential Issue-vote Engine

이 디렉터리는 재구성된 pre-2025 baseline의 대통령선거 엔진이다. 평가 범위는
`pres_2002`부터 `pres_2022`까지이며, rolling-origin에서 `pres_1997`은 2002 예측을
위한 warmup으로만 사용한다.

```powershell
python presidential_issue_engine\robustness_check.py
python presidential_issue_engine\audit_point_in_time.py --deep
```

2025 결과 평가 모듈과 2025 성능을 포함한 gain 선택 스크립트는 이 clean 폴더에
없다. 사후 선택 레이어는 기본 비활성화되어 있다. 자세한 출처와 제한은
`docs/PRE2025_RECONSTRUCTION.md`를 따른다.
