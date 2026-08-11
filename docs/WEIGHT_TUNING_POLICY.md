# Weight Tuning Policy

## Fixed Sample

- rolling warmup only: `pres_1997`
- scored and weight-selection elections: `pres_2002`, `pres_2007`, `pres_2012`,
  `pres_2017`, `pres_2022`
- latest allowed outcome label: `pres_2022`

## Prohibited Uses

- 이후 선거 결과를 가중치, 임계값, 변수 선택에 사용하지 않는다.
- 이후 선거를 holdout, 참고표, ablation, sanity check로 비교하지 않는다.
- 보존 폴더의 배율이나 평가표를 활성 엔진으로 복사하지 않는다.
- 선거별 수동 이슈 seed를 forecast 입력으로 사용하지 않는다.
- 자동 이슈 seed 생성기는 득표수·득표율·당선 여부를 읽지 않는다.

## Selection Protocol

레이어 재선택은 Ridge alpha, sparse residual calibration, neutral context, overlay,
conversion, district terrain, regionalism, within-bloc regional transfer,
third-candidate 순서의 고정 coordinate search를 사용한다.
outer 목표 선거의 설정은 그보다 앞선 scored 선거로만
고른다. 과거 scored 선거가 두 개 미만이면 중립 설정을 유지한다.

작은 표본에서 우연한 개선을 채택하지 않도록 필요한 최소 개선폭을 미리 고정한다.

- 과거 선거 2개: `0.10%p`
- 과거 선거 3개: `0.075%p`
- 과거 선거 4개 이상: `0.05%p`

Within-bloc transfer는 사전 activation이 `0.001` 이상인 과거 선거가 최소 두 개일
때만 scale 선택을 허용하고, 최대값을 `0.50`으로 제한한다.

최종 배포 설정은 2002~2022 전체 rolling으로 고르지만, 그 수치는 외부 검증으로
해석하지 않는다. 모델 선택 성능은 outer fold별 설정이 분리된 nested rolling으로
보고한다.

```powershell
python scripts\build_through2022_automatic_issue_seeds.py
python scripts\rederive_layers_through2022.py
python presidential_issue_engine\audit_weight_selection_boundary.py
```
