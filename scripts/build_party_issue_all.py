"""5개 대선 전부 → (슬롯 × 이슈) 정당-이슈 연결 (회의록 + 명부, 일회성).

각 대선의 선거창은 특정 국회 대수에 들어간다(날짜 기준):
  2002→16대, 2007→17대, 2012→19대, 2017→20대, 2022→21대.
그 대수의 회의록을 명부({의원:정당진영})로 재처리해 (정당진영×이슈) 발언을 세고,
개표결과의 슬롯 정당(party_name)으로 진영→슬롯 매핑한다.
"""

from __future__ import annotations

import io
import sys
import zipfile
from collections import Counter

import pandas as pd

sys.path.insert(0, "src")
from news_collector.sources.assembly_batch import ELECTION_WINDOWS, accumulate, iter_rows_from_xlsx  # noqa: E402
from news_collector.sources.datalab import load_issue_keywords  # noqa: E402
from news_collector.sources.member_party import party_bloc  # noqa: E402

MASTER = r"C:/Users/최성준/Downloads/trash_dataset.zip"
RESULTS = "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
ELECTION_TO_ASSEMBLY = {"pres_2002": "16", "pres_2007": "17", "pres_2012": "19", "pres_2017": "20", "pres_2022": "21"}


def main() -> None:
    km = load_issue_keywords("presidential_issue_engine/fixed_dataset/issue_keywords.csv")
    roster = pd.read_csv("data/assembly_roster.csv", dtype=str)
    results = pd.read_csv(RESULTS)
    z = zipfile.ZipFile(MASTER)

    all_rows = []
    for eid, assembly in ELECTION_TO_ASSEMBLY.items():
        # 1) 이 대수 명부 → {이름: 진영}
        rdf = roster[roster["daesu"] == assembly]
        member_to_bloc = {n: b for n, b in zip(rdf["name"], rdf["bloc"]) if n and b}
        # 2) 개표결과 슬롯 정당 → 진영 → {진영: 슬롯}
        rr = results[(results.election_id == eid) & (results.slot.isin(["A", "B", "C"]))]
        bloc_to_slot = {}
        for r in rr.itertuples():
            if r.slot == "C" and not r.is_active_slot:
                continue
            bloc_to_slot[party_bloc(r.party_name)] = r.slot
        # 3) 이 대수 회의록 처리 (선거창 필터)
        window = {eid: ELECTION_WINDOWS[eid]}
        mem: Counter = Counter()
        targets = [i for i in z.infolist() if f"제{assembly}대" in i.filename and not i.is_dir()]
        for info in targets:
            if info.filename.lower().endswith(".xlsx"):
                accumulate(iter_rows_from_xlsx(z.read(info)), km, window, member_to_bloc, Counter(), mem)
            elif info.filename.lower().endswith(".zip"):
                inner = zipfile.ZipFile(io.BytesIO(z.read(info)))
                for j in inner.infolist():
                    if j.filename.lower().endswith(".xlsx"):
                        accumulate(iter_rows_from_xlsx(inner.read(j)), km, window, member_to_bloc, Counter(), mem)
        # 4) 진영 → 슬롯
        for (e, bloc, issue, wk), c in mem.items():
            all_rows.append({"election_id": e, "slot": bloc_to_slot.get(bloc, "alpha"), "issue_name": issue, "mentions": c})
        print(f"{eid} ({assembly}대): mem={len(mem)} | 슬롯정당={bloc_to_slot}", flush=True)

    df = pd.DataFrame(all_rows).groupby(["election_id", "slot", "issue_name"], as_index=False)["mentions"].sum()
    df["emphasis_volume"] = df.groupby("election_id")["mentions"].transform(lambda s: (s / s.max()).round(4))
    df["emphasis_within"] = df.groupby(["election_id", "slot"])["mentions"].transform(lambda s: (s / s.sum()).round(4))
    out = "data/candidate_issue_link.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"DONE: {len(df)} rows, {df.election_id.nunique()} elections → {out}", flush=True)


if __name__ == "__main__":
    main()
