# Through-2022 Automatic Issue Seed

## Forecast Rule

선거별 이슈 방향과 후보 귀속을 사람이 입력하지 않는다. 자동 생성기는 선거 전에
사용 가능한 국회 발언 집계만 읽으며 득표수, 득표율, 당선 여부를 입력으로 사용하지
않는다.

```powershell
python scripts\build_through2022_automatic_issue_seeds.py
```

향후 선거는 해당 선거의 사전시점 집계가 아래 입력 파일에 준비된 뒤 선거 ID만
지정한다. 선거 ID는 처리 범위를 고르는 운영 입력일 뿐, 이슈명·방향·강도·후보
귀속은 입력하지 않는다.

```powershell
python scripts\build_through2022_automatic_issue_seeds.py --elections pres_2027
```

필수 사전 집계가 없는 선거 ID는 빈 seed를 만들지 않고 오류로 중단한다.

## Inputs

- `data/issue_salience_assembly.csv`
- `data/candidate_issue_link.csv`
- `data/raw/assembly_issue_character_overlay.csv`
- `data/raw/candidate_public_treatment.csv`
- `data/raw/candidate_party_tone_gap.csv`

후보명과 슬롯도 개표 결과가 아니라 `candidate_public_treatment.csv`에서 가져온다.

## Generated Signals

후보-이슈 연계 강도는 후보별 발언 내 이슈 비중 순위 60%, 전국 이슈 강도 순위
20%, 후보 연결 신뢰도 20%로 계산한다. 방향은 이슈 문장 방향 70%와 후보에 대한
자당·타당 및 공적 대우 신호 30%를 결합한 뒤 `tanh`로 -1~1에 제한한다.

전국 이슈축은 선거별 누적 salience 상위 두 이슈로 만든다. 후보 귀속은 자동
후보-이슈 연계와 연속 방향 신호에서 파생한다. 모든 값은 고정식으로 계산되며
선거별 수동 수정 단계가 없다.

## Outputs

- `data/raw/auto_issue_seed/candidate_issue_profile.csv`
- `data/raw/auto_issue_seed/mega_issue_axis.csv`
- `data/raw/auto_issue_seed/mega_issue_attribution.csv`

현재 생성 행은 각각 48, 10, 18행이다. 수동 seed 세 파일은 보존만 하며 활성
forecast 입력으로 읽지 않는다. 활성 상태는
`data/config/through2022_rederived_layers.json`의 registry가 결정한다.

## Validation

- scored scope: 2002, 2007, 2012, 2017, 2022
- 각 행의 `available_date`: 해당 선거 D-1 이하
- strict outcome-invariance: 215/215
- 동일 가중치 seed-off rolling: `3.763840%p`
- automatic seed-on rolling: `3.763840%p`

현재 선택 설정에서는 자동 seed의 추가행이 최종 predictor를 바꾸지 않아 두 값이
같다. 이 비교는 자동 seed 효과의 진단이며 공식을 결과에 맞춰 재조정하는 용도로
사용하지 않는다.
