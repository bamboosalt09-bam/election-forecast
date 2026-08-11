# Through-2022 Rederived Scope

- scored elections: 2002, 2007, 2012, 2017, 2022
- rolling warmup: 1997
- Ridge alpha: 1.20
- sparse region residual calibration: off
- neutral issue-context scale: 0.60
- district-election terrain adjustment: off, scale 0.0
- selected regionalism: scale 0.15, anchor 1.0
- within-bloc regional transfer: scale 0.50, reservoir 1.00, stronghold 0.25
- selected third-candidate competitiveness gate: on
- selected third-candidate character multiplier: on
- manual issue seed: off
- automatic Assembly issue seed: on
- electorate issue sensitivity: integrated, active preference gain 0.04 from capped nested selection
- historical fixed post-hoc electorate experiment: preference gain 0.04
- electorate terrain anchor: off
- turnout and nonvoter response: off pending official history
- rejected layers: issue-character overlay, candidate vote-conversion adjustment,
  district-election terrain adjustment

Verified snapshot:

- frozen baseline contest-vote weighted macro MAE: 4.628%p
- capped strict nested electorate result: 4.614%p
- uncapped nested preference-gain experiment: 5.471%p, rejected
- non-active fixed structural experiment: 4.594%p
- active rolling row MAE: 3.710%p
- active rolling contest-vote weighted macro MAE: 3.718%p
- LOEO row MAE: 4.98%p
- R2: 0.876
- strict deep PIT audit: PASS, 215 rows
- pytest: see `docs/HANDOFF_CURRENT_STATE.md` for the latest complete run

선택 표본 rolling과 nested rolling을 같은 지표로 부르지 않는다. 모델 선택의 더
엄격한 성능 판단에는 nested rolling 값을 사용한다.
