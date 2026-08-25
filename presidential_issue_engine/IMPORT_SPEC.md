<!-- active-model-version: v31 -->
# 입력 명세 — 활성 모델이 실제로 먹는 자료

이 문서는 **활성 모델이 실제로 읽는 입력**만 기술한다. 저장소에는 쓰이지 않는
임포터가 몇 개 남아 있고, 그것들은 아래 "활성 경로에 없는 것"에 분리해 적었다.

권리·재배포 판단은 `docs/PUBLIC_DATA_SOURCES.json`과
`docs/DATA_PROVENANCE_AND_REDISTRIBUTION.md`가 정본이다. 이 문서는 형식 명세다.

---

## 1. 이슈 부각도 (salience) — 국회 회의록 발화

**활성 파일**: `data/issue_salience_assembly.csv`
**계측**: `instrument = assembly_speech` — **1001행 전부**. 다른 계측값은 없다.

| 컬럼 | 필수 | 용도 |
| --- | --- | --- |
| `election_id` | ✅ | 선거 구분 |
| `issue_name` | ✅ | 이슈 축 |
| `period` | ✅ | 주 단위 버킷 |
| `raw_value` | ✅ | 원 빈도 |
| `salience_score` | ✅ | 선거 내 정규화값 |
| `instrument` | ✅ | 출처 추적 (`assembly_speech`) |
| `available_date` | ✅ | PIT 감사 기준일 |

**출처**: 국회 회의록 / 열린국회정보 공개 데이터. 원문 말뭉치는 배포하지 않고
파생 집계만 싣는다 — 제공처 약관이 제공 정보의 무변경 공유를 금지한다.

## 2. 선거 결과 (종속변수)

중앙선관위 시도별 득표 → `presidential_results_standardized.csv`
(빈 템플릿: `fixed_dataset/templates/`).

## 3. 지역 투표량 (예측시점 가중)

`fixed_dataset/pres_1997_regional_turnout.csv` 외에는 결과표에서 직접 읽는다.
V30 이후 종단 변환은 **직전 선거**의 지역 유효투표수로 가중한다.

---

## 활성 경로에 없는 것

아래 임포터는 코드로 남아 있으나 **활성 모델은 한 번도 읽지 않는다.** 산출물도,
데이터 행도, 입력 매니페스트 등재도 없다.

| 임포터 | 계측 이름 | 활성 데이터 행 |
| --- | --- | ---: |
| `src/news_collector/sources/bigkinds_metadata.py` | `bigkinds_meta` | **0** |
| `src/news_collector/sources/bigkinds_salience.py` | `bigkinds_count` | **0** |
| 연합 제목 크롤 경로 | `yonhap_title_count` | **0** |
| 네이버 데이터랩 경로 | `datalab_search` | **0** |

이 문서의 이전 판은 BIGKinds 뉴스 메타데이터를 **1번 항목**으로 두어, 마치 활성
모델이 뉴스 메타데이터로 이슈 부각도를 만드는 것처럼 기술했다. 그런 적이 없다.
경위는 `docs/DIAGNOSIS_IMPORT_SPEC_DRIFT_20260825.md`에 있다.

임포터를 지우지 않는 이유는 그것들이 초기 설계의 기록이기 때문이고, 지우면 그
기록이 사라진다. 대신 활성 여부를 여기에 명시한다.

---

## 실제 흐름

```
국회 회의록 발화 ─► (의원 × 이슈) 빈도 ─► issue_salience_assembly.csv
                                              (instrument=assembly_speech)
                                                        │
                                                        ▼
                          후보·정당·지역 자료와 결합 ─► PIT 감사 ─► 중첩 Ridge
```

모든 salience 행에 `instrument`가 박혀 어느 수치가 어디서 왔는지 추적된다. 현재
그 값은 `assembly_speech` 하나뿐이며, 그것이 이 모델의 유일한 부각도 출처다.
