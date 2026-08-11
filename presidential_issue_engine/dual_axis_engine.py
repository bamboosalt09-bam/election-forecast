"""두 축(지역 + 세대) 동시 분석 — 이슈가 어느 축의 표심을 더 설명하나?

지역축: 이슈우위(전국값) → 지역 표심 (지역주의 때문에 약할 것으로 예상)
세대축: 이슈우위를 '청년성향 × 세대'로 변조 → 세대 표심 (이슈-세대 연결이 강해 더 클 것)

각 축마다: M0(구도) → M1(+이슈) ΔR², 슬롯 통제 후 ΔR²(robustness), 교차검증 %p.
다층모형·softmax 없이 numpy 다중회귀. AI 미사용.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from issue_vote_engine import ORDER, assemble as assemble_region, loeo_cv, ols  # noqa: E402
from news_collector.sources.member_party import party_bloc  # noqa: E402

RESULTS = "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
SALIENCE = "data/issue_salience_assembly.csv"
LINK = "data/candidate_issue_link.csv"
GENVOTE = "data/generation_vote.csv"
GENSENS = "presidential_issue_engine/fixed_dataset/generation_issue_sensitivity.csv"
YOUTHNESS = {"20": 1.0, "30": 0.5, "40": 0.0, "50": -0.5, "60+": -1.0}


def assemble_generation() -> pd.DataFrame:
    res = pd.read_csv(RESULTS)
    res["bloc"] = res["party_name"].map(party_bloc)
    sb = res.groupby(["election_id", "slot"])["bloc"].first().reset_index()  # 선거×슬롯 → 진영

    sal = pd.read_csv(SALIENCE)
    natsal = sal.groupby(["election_id", "issue_name"])["salience_score"].mean().reset_index(name="salience")
    link = pd.read_csv(LINK)
    lean = pd.read_csv(GENSENS)[["issue_name", "youth_lean"]]

    # 선거×슬롯별 youth_issue = Σ_이슈 salience × 소유 × 청년성향
    m = link.merge(natsal, on=["election_id", "issue_name"]).merge(lean, on="issue_name", how="left").fillna({"youth_lean": 0.0})
    m["c"] = m["emphasis_within"] * m["salience"] * m["youth_lean"]
    youth_issue = m.groupby(["election_id", "slot"])["c"].sum().reset_index(name="youth_issue")

    # 세대 득표 (퍼센트→분수, alpha 보강)
    gv = pd.read_csv(GENVOTE)
    piv = gv.pivot_table(index=["election_id", "generation"], columns="slot", values="vote_share_pct", aggfunc="sum").fillna(0)
    piv["alpha"] = 100 - piv.sum(axis=1)
    gvl = piv.reset_index().melt(id_vars=["election_id", "generation"], var_name="slot", value_name="vs")
    gvl["vote_share"] = gvl["vs"] / 100
    gvl = gvl.merge(sb, on=["election_id", "slot"], how="left")
    gvl["eidx"] = gvl["election_id"].map({e: i for i, e in enumerate(ORDER)})

    # 진영별 세대 득표 (gen_base용)
    bloc_gv = gvl.groupby(["election_id", "generation", "bloc"])["vote_share"].sum().reset_index()

    rows = []
    yi = youth_issue.set_index(["election_id", "slot"])["youth_issue"].to_dict()
    bg = {(e, g, b): v for e, g, b, v in bloc_gv.itertuples(index=False)}
    for r in gvl.itertuples():
        if r.slot == "alpha" or pd.isna(r.eidx) or r.eidx == 0:
            continue
        prev = ORDER[int(r.eidx) - 1]
        gen_base = bg.get((prev, r.generation, r.bloc), np.nan)
        if pd.isna(gen_base):
            continue
        gif = yi.get((r.election_id, r.slot), 0.0) * YOUTHNESS.get(str(r.generation), 0.0)
        rows.append({"election_id": r.election_id, "generation": r.generation, "slot": r.slot,
                     "vote_share": r.vote_share, "gen_base": gen_base, "gen_issue_fit": gif})
    return pd.DataFrame(rows)


def analyze(df, base_cols, issue_col, label):
    bc = base_cols
    _, r0, _, _ = ols(df[bc].to_numpy(float), df["vote_share"].to_numpy(float))
    _, r1, _, _ = ols(df[bc + [issue_col]].to_numpy(float), df["vote_share"].to_numpy(float))
    # 슬롯 통제 robustness
    df = df.copy()
    df["sA"] = (df.slot == "A").astype(float)
    df["sB"] = (df.slot == "B").astype(float)
    _, rs, _, _ = ols(df[["sA", "sB"] + bc].to_numpy(float), df["vote_share"].to_numpy(float))
    _, rsi, _, _ = ols(df[["sA", "sB"] + bc + [issue_col]].to_numpy(float), df["vote_share"].to_numpy(float))
    cv0 = loeo_cv(df, ["sA", "sB"] + bc)
    cv1 = loeo_cv(df, ["sA", "sB"] + bc + [issue_col])
    print(f"=== {label} ({len(df)}행) ===")
    print(f"  M0(구도)        R²={r0:.3f}   →   M1(+이슈) R²={r1:.3f}   |   소박한 ΔR²={r1-r0:+.3f}")
    print(f"  [robust] 슬롯+구도 R²={rs:.3f} → +이슈 R²={rsi:.3f} | 통제 후 ΔR²={rsi-rs:+.3f}  ← 진짜 효과")
    print(f"  [robust] CV %p: 슬롯+구도 {cv0:.2f} → +이슈 {cv1:.2f}  (개선 {cv0-cv1:+.2f}%p)\n")


def main() -> None:
    print("● 지역축\n")
    reg = assemble_region()
    analyze(reg, ["regional_base"], "issue_advantage", "지역 표심")
    print("● 세대축\n")
    gen = assemble_generation()
    analyze(gen, ["gen_base"], "gen_issue_fit", "세대 표심")


if __name__ == "__main__":
    main()
