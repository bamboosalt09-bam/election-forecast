# 가져올 자료 명세 (사용자 export → 임포터가 바로 먹는 형식)

당신이 직접 가져오는 자료가 임포터에 바로 들어가도록 **필요한 컬럼·형식**을 못박는다.
모두 **메타데이터/공공기록**이라 합법(본문 저장 없음).

---

## 1. BIGKinds 뉴스 메타데이터 (과거 포함 salience + 인물 연결)

**어디서**: BIGKinds(bigkinds.or.kr) → 뉴스분석 → 기간/조건 설정 → **데이터 다운로드(엑셀)**.
본문 말고 **메타데이터**만 받으면 됨.

**기간**: 대선별로 분리 (예: 18대 → 2012-09-01 ~ 2012-12-19). 선거당 1개 파일.

**필요 컬럼** (BIGKinds 표준 export 컬럼명을 그대로 인식):

| 컬럼 | 필수 | 용도 |
| --- | --- | --- |
| `일자` | ✅ | 주 단위 버킷 (YYYYMMDD/YYYY-MM-DD 다 됨) |
| `제목` | ✅ | 이슈 키워드 매칭 → salience |
| `키워드` | 권장 | 매칭 정확도↑ (쉼표 구분) |
| `인물` | 권장 | 후보 동시언급 → candidate_link (쉼표 구분) |
| `언론사`, `통합 분류1` | 선택 | 참고/필터 |

→ 임포터: `bigkinds_metadata.metadata_to_salience` (instrument=`bigkinds_meta`) +
`metadata_to_candidate_issue`. 컬럼명이 다르면 매핑만 주면 됨.

**주의**: 검색을 *이슈별로* 좁히지 말고 **그 선거 기간의 정치/사회 뉴스 메타데이터를
넓게** 받으면, 키워드 매칭은 우리 사전(`issue_keywords.csv`)으로 로컬에서 한다.
(그래야 키워드 선택 편향이 수집 단계에 안 들어감)

---

## 2. 국회의원 발언 목록 (direction·이슈 소유 — 공공기록)

**어디서**: 열린국회정보(open.assembly.go.kr) API / 국회 회의록 등 공개 데이터.

**필요 컬럼** (롱 CSV 한 장; 컬럼명은 아래로 맞춰주면 좋음):

| 컬럼 | 필수 | 용도 |
| --- | --- | --- |
| `speaker` (의원명) | ✅ | 후보/슬롯 매칭 |
| `party` (정당) | 권장 | 진영·direction 보조 |
| `date` (일자) | ✅ | 주 버킷 |
| `text` (발언내용/요지) 또는 `topic` | ✅ | 이슈 키워드 매칭 |
| `meeting` (회의명) | 선택 | 참고 |

→ 용도: (의원 × 이슈) 발언 빈도 = **그 정치인이 어떤 이슈를 얼마나 미는가(이슈 소유)**
= candidate_link/strength 신호. **방향(호오)** 은 별도 코딩(정당 입장 + 사설 정독).
*(이 임포터는 형식 확정되면 BIGKinds 것과 같은 패턴으로 바로 만든다.)*

---

## 3. 선거 결과 (정답지 — 별도)

중앙선관위 시도별 득표 → `presidential_results_standardized.csv`
(템플릿: `fixed_dataset/templates/`). salience·발언과 상관·회귀로 비교할 **종속변수**.

---

## 처리 후 자동 흐름

```
연합 제목(크롤, 최근)        ┐
BIGKinds 메타(당신, 과거포함) ├─► salience (instrument별 provenance) ─┐
                             ┘                                        │
국회 발언(당신, 공공) ─► (의원×이슈) candidate_link ─────────────────┤
사설 정독(당신) ─► direction 코딩 ──────────────────────────────────┤
                                                                      ▼
                                        issue_events → rollup → 상관·회귀(R²)·%P
```

모든 salience 행에 `instrument`(yonhap_title_count / bigkinds_meta / datalab_search)가
박혀, "어느 수치가 어디서 왔는지" 추적된다.
