"""21대 회의록 → (정당 진영 × 이슈) 발언 강조 → 2022 슬롯-이슈 연결 (일회성).

발언자→정당(진영) 매핑으로 회의록을 재처리해, 2022 대선 슬롯 A/B/C의 정당이
어떤 이슈를 얼마나 다뤘는지(이슈 소유/연결)를 산출한다.
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
from news_collector.sources.member_party import load_member_party, party_bloc  # noqa: E402
from news_collector.sources.salience_base import normalize_within_election  # noqa: E402

MASTER = r"C:/Users/최성준/Downloads/trash_dataset.zip"
ROSTER = r"C:/Users/최성준/Downloads/역대국회의원인적사항.csv"
WINDOWS = {"pres_2022": ELECTION_WINDOWS["pres_2022"]}
# 2022 대선 슬롯 ↔ 정당 진영 (개표결과: A=윤석열 국민의힘, B=이재명 민주, C=심상정 정의)
BLOC_TO_SLOT = {"국민의힘": "A", "더불어민주당": "B", "정의당": "C"}


def main() -> None:
    km = load_issue_keywords("presidential_issue_engine/fixed_dataset/issue_keywords.csv")
    name_party = load_member_party(ROSTER, term="21")
    member_to_bloc = {name: party_bloc(party) for name, party in name_party.items()}
    print(f"명부 {len(member_to_bloc)}명 | 진영 분포: {Counter(member_to_bloc.values()).most_common(5)}", flush=True)

    sal: Counter = Counter()
    mem: Counter = Counter()  # (election, bloc, issue, week)
    z = zipfile.ZipFile(MASTER)
    targets = [i for i in z.infolist() if "21대" in i.filename and not i.is_dir()]
    for idx, info in enumerate(targets, 1):
        if info.filename.lower().endswith(".xlsx"):
            accumulate(iter_rows_from_xlsx(z.read(info)), km, WINDOWS, member_to_bloc, sal, mem)
        elif info.filename.lower().endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(z.read(info)))
            for j in inner.infolist():
                if j.filename.lower().endswith(".xlsx"):
                    accumulate(iter_rows_from_xlsx(inner.read(j)), km, WINDOWS, member_to_bloc, sal, mem)
        print(f"[{idx}/{len(targets)}] {info.filename.split('/')[-1][:42]} | mem keys={len(mem)}", flush=True)

    # (bloc, issue) 합산 → 슬롯 매핑 → 선거 내 정규화
    rows = []
    for (eid, bloc, issue, wk), c in mem.items():
        slot = BLOC_TO_SLOT.get(bloc, "alpha")
        rows.append({"election_id": eid, "slot": slot, "issue_name": issue, "mentions": c})
    df = pd.DataFrame(rows).groupby(["election_id", "slot", "issue_name"], as_index=False)["mentions"].sum()
    # 두 지표: 절대량(당 규모 반영) vs 당내비중(이슈 소유 프로파일 — 권장)
    df["emphasis_volume"] = df.groupby("election_id")["mentions"].transform(lambda s: (s / s.max()).round(4))
    df["emphasis_within"] = df.groupby("slot")["mentions"].transform(lambda s: (s / s.sum()).round(4))
    out = "data/candidate_issue_link.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"DONE: {len(df)} (슬롯×이슈) rows → {out}", flush=True)
    # 슬롯별 상위 이슈 미리보기
    for slot in ("A", "B", "C"):
        top = df[df.slot == slot].sort_values("mentions", ascending=False).head(4)
        print(f"  슬롯{slot}: " + ", ".join(f"{r.issue_name}({int(r.mentions)})" for r in top.itertuples()), flush=True)


if __name__ == "__main__":
    main()
