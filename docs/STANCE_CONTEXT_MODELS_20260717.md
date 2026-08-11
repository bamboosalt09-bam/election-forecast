# 국회 발언 문맥 모델 추가 실험

## 결론

국회도서관 자료 연계 계층과 두 개의 추가 문맥 인코더를 구현했다. 그러나 새
인코더는 잠금 감사에서 활성화 기준을 통과하지 못했다. 현재 forecast 입력,
이슈 가중치, 득표율 산식에는 연결하지 않는다.

- KLUE 후보: `klue/roberta-small`
- NLI 후보: `jhgan/ko-sroberta-nli`
- 결합 후보: 기존 target-aware v8 + KLUE + NLI의 2-of-3 다수결
- 데이터 범위: `pres_2002`부터 `pres_2022`까지만 허용
- 선거 결과 변수 사용: 없음
- 2025 자료 사용: 없음
- 활성 forecast 변경: 없음

## 국회도서관의 역할

국회도서관은 이 프로젝트에 바로 넣을 수 있는 긍정/중립/부정 분류기를
제공하는 곳이 아니라 회의록과 의정자료를 제공하는 도메인 자료원이다.

- 국회도서관 Open API: https://www.nanet.go.kr/usermadang/etc/openApiView.do
- 국가학술정보 클라우드: https://losi-open.nanet.go.kr/user/information.do

따라서 자료원과 분류 모델을 분리했다. 국회도서관 export/API 결과는
`election_id,available_date,text` 형식으로 정규화한 뒤
`scripts/import_nanet_context_corpus.py`를 통과해야 한다. 이 검증기는 다음을
거부한다.

1. 선거별 기준일 이후 문장
2. 2022 대선 이후 문장
3. 득표율, 개표결과, 당선자, 실제 마진 등 결과형 열
4. 알 수 없는 선거 ID 또는 잘못된 날짜

API 키나 정규화된 export가 제공되지 않은 상태에서는 원격 스키마를 추정해서
가져오지 않는다. 특히 22대 국회 자료는 through-2022 학습 경계를 넘으므로 이
실험의 scored classifier에는 사용할 수 없다. 등록 상태는
`data/shadow/context_model_sources.json`에 기록한다.

## 모델 고정 정보

| 후보 | revision | 인코딩 |
|---|---|---|
| KLUE RoBERTa small | `b6b4c36d827e0293ae2fcf04d527072f10a23064` | 최대 256토큰 mean pooling |
| Ko-SRoBERTa NLI | `c4e15f24df2aceadfc931e2a57094726b2409861` | sentence-transformers normalized embedding |

두 인코더는 fine-tuning하지 않았다. 고정 임베딩 위에 3-class logistic head만
학습했다. 입력은 후보/정당 이름을 마스킹한 현재 문장과 위험 플래그가 허용하는
앞뒤 문맥이다. 개발 자료는 이미 검토된 잠금 감사 v1 80행과 v2 78행이며,
추가 3,290행은 낮은 가중치의 보조 자료다.

## 개발 결과

개발 자료는 모델과 임계값 선택에 사용했으므로 독립 성능이 아니다.

| 후보 | 개발행 | 방향 출력 | 유해 오판 | 방향 커버리지 |
|---|---:|---:|---:|---:|
| KLUE v9 | 158 | 4 | 0 | 4.17% |
| NLI v10 | 158 | 5 | 0 | 5.21% |

유해 오판을 0으로 맞추면 두 모델 모두 거의 모든 문장에 중립으로 기권한다.
따라서 단독 분류기로 충분하지 않다.

## 5,000문장 shadow 적용

| 후보 | 방향 출력 | 음성 | 양성 |
|---|---:|---:|---:|
| 기존 target-aware v8 | 107 | 105 | 2 |
| KLUE v9 | 25 | 23 | 2 |
| NLI v10 | 84 | 75 | 9 |
| 2-of-3 consensus v11 | 23 | 22 | 1 |

세 자식 모델 사이에 양성/음성 충돌은 관찰되지 않았다. 그러나 이는 정밀도가
높다는 뜻이 아니다. 세 모델이 같은 약한 라벨 구조에서 비슷한 잘못을 반복할
수 있기 때문이다.

## 잠금 감사

NLI가 새로 출력한 미검토 문장으로 v4를 잠근 뒤 40건 모두를 문맥과 함께
검토했다.

| 항목 | NLI v10 독립 v4 감사 |
|---|---:|
| 감사 방향 출력 | 40 |
| 올바른 방향 출력 | 27 |
| 중립을 방향으로 오판 | 13 |
| 방향 부호 반전 | 0 |
| 방향 정밀도 | 67.5% |
| 유해 오류율 | 32.5% |
| 유해 오류율 단측 95% 상한 | 46.63% |

2-of-3 consensus 23건 중 20건은 이미 v1/v2 개발 감사에서 본 문장이었다. 새로
남은 v3 감사 3건에서는 1건만 맞고 2건이 중립 오판이었다. 표본이 작지만 채택을
거부하기에는 충분하다.

주요 오류는 다음과 같다.

1. 외부 전문가, 언론, 여론조사의 평가를 화자 자신의 입장으로 귀속
2. 질문을 확정적 비판으로 변환
3. 정부가 제3의 정책을 지지한다는 서술을 정부에 대한 지지로 변환
4. 문장에 함께 나온 다른 정부/인물을 잘못 target으로 선택
5. 뒤 문장에서 반박되는 주장을 현재 문장만 보고 채택

## 채택 판단

40% 커버리지는 채택 기준이 아니다. 현재 기준은 독립 방향 출력 59건 이상,
관찰 유해 오류 0건, 유해 오류율 단측 95% 상한 5% 이하, target 귀속 감사,
PIT 감사, strict rolling 비열화다.

KLUE/NLI/consensus는 이 기준을 통과하지 못했다. 다음 개선은 모델 수를 더 늘리는
것이 아니라 아래 순서의 계층 분류가 필요하다.

1. target이 실제 평가 대상인지 판별
2. 발언 소유자가 화자인지, 인용된 외부 주체인지 판별
3. 질문/보고/반박/가정 문장을 중립 처리
4. 위 세 단계를 통과한 문장에만 양성/음성 방향 판정

## 재현

```powershell
.\.venv-stance\Scripts\python.exe scripts\train_stance_context_encoder.py --encoder klue-small
.\.venv-stance\Scripts\python.exe scripts\train_stance_context_encoder.py --encoder ko-nli
.\.venv-stance\Scripts\python.exe scripts\apply_stance_context_encoder.py --artifact outputs\assembly_stance\stance_klue_context_v9\stance_klue_context_v9.joblib --output-dir outputs\assembly_stance\stance_klue_context_v9\application_5000 --local-files-only
.\.venv-stance\Scripts\python.exe scripts\apply_stance_context_encoder.py --artifact outputs\assembly_stance\stance_ko_nli_context_v10\stance_ko_nli_context_v10.joblib --output-dir outputs\assembly_stance\stance_ko_nli_context_v10\application_5000 --local-files-only
.\.venv-stance\Scripts\python.exe scripts\build_stance_context_consensus.py
```

감사 원본과 라벨은 `data/shadow/stance_locked_audit_v3*.csv` 및
`data/shadow/stance_locked_audit_v4*.csv`에, 지표는 각 모델 output 폴더의
`locked_audit_*_metrics.json`에 있다.
