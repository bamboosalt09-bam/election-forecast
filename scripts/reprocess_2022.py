"""21대 국회 회의록 → pres_2022 salience만 재처리 후 기존 CSV에 append (일회성)."""

from __future__ import annotations

import io
import sys
import zipfile
from collections import Counter

import pandas as pd

sys.path.insert(0, "src")
from news_collector.sources.assembly_batch import ELECTION_WINDOWS, accumulate, iter_rows_from_xlsx  # noqa: E402
from news_collector.sources.datalab import load_issue_keywords  # noqa: E402
from news_collector.sources.salience_base import normalize_within_election  # noqa: E402

MASTER = r"C:/Users/최성준/Downloads/trash_dataset.zip"
CSV = "data/issue_salience_assembly.csv"
WINDOWS = {"pres_2022": ELECTION_WINDOWS["pres_2022"]}  # 2022만


def main() -> None:
    km = load_issue_keywords("presidential_issue_engine/fixed_dataset/issue_keywords.csv")
    sal: Counter = Counter()
    z = zipfile.ZipFile(MASTER)
    targets = [i for i in z.infolist() if "21대" in i.filename and not i.is_dir()]
    for idx, info in enumerate(targets, 1):
        name = info.filename.split("/")[-1]
        if info.filename.lower().endswith(".xlsx"):
            accumulate(iter_rows_from_xlsx(z.read(info)), km, WINDOWS, None, sal, Counter())
        elif info.filename.lower().endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(z.read(info)))
            for j in inner.infolist():
                if j.filename.lower().endswith(".xlsx"):
                    accumulate(iter_rows_from_xlsx(inner.read(j)), km, WINDOWS, None, sal, Counter())
        print(f"[{idx}/{len(targets)}] {name[:45]} | keys={len(sal)}", flush=True)

    rows = [{"issue_name": i, "period": w, "count": c} for (e, i, w), c in sal.items()]
    new_sal = normalize_within_election(pd.DataFrame(rows), "count", "pres_2022", "assembly_speech")
    prev = pd.read_csv(CSV)
    prev = prev[prev["election_id"] != "pres_2022"]
    out = pd.concat([prev, new_sal], ignore_index=True)
    out.to_csv(CSV, index=False, encoding="utf-8-sig")
    print(f"DONE: pres_2022 {len(new_sal)} rows appended → {CSV} (total {len(out)})", flush=True)


if __name__ == "__main__":
    main()
