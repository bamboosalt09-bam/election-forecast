"""이슈를 테마로 분해해 지역·세대 매개효과를 개별로 본다.

한 덩어리(issue_advantage)로 뭉치면 어떤 이슈가 작용하는지 사라진다. 19개를 동시에
넣으면 과적합(세대 N=35)이라, **테마 5개**로 묶어:
  - 지역: 각 테마가 지역 표심에 주는 ΔR² (구도 통제, 한 테마씩)
  - 세대: 각 테마 × 세대(청년성향) 상호작용의 ΔR² (세대 통제) = '이 테마가 *맞는 세대*를
    얼마나 움직이나' 매개관계를 데이터로 추정
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dual_axis_engine import assemble_generation, YOUTHNESS  # noqa: E402
from issue_vote_engine import LINK, SALIENCE, assemble as assemble_region, ols  # noqa: E402

THEMES = {
    "경제민생": ["economy_growth", "jobs_labor", "inflation_livelihood", "housing"],
    "복지교육": ["welfare_pension", "education"],
    "안보외교": ["security_nk", "foreign_policy"],
    "사회젠더": ["gender_generation", "regional_dev"],
    "도덕성": ["corruption_integrity", "family_legal_risk", "gaffe_event"],
}


def theme_features() -> pd.DataFrame:
    """(election, slot)별 테마 신호 = Σ_{이슈∈테마} salience × 이슈소유."""
    sal = pd.read_csv(SALIENCE).groupby(["election_id", "issue_name"])["salience_score"].mean().reset_index(name="sal")
    link = pd.read_csv(LINK)
    m = link.merge(sal, on=["election_id", "issue_name"], how="left").fillna({"sal": 0.0})
    m["sig"] = m["emphasis_within"] * m["sal"]
    issue2theme = {i: t for t, lst in THEMES.items() for i in lst}
    m["theme"] = m["issue_name"].map(issue2theme)
    m = m.dropna(subset=["theme"])
    return m.groupby(["election_id", "slot", "theme"])["sig"].sum().reset_index().pivot_table(
        index=["election_id", "slot"], columns="theme", values="sig", aggfunc="sum"
    ).fillna(0).reset_index()


def dr2(df, base_cols, add_cols):
    _, r0, _, _ = ols(df[base_cols].to_numpy(float), df["vote_share"].to_numpy(float))
    _, r1, _, _ = ols(df[base_cols + add_cols].to_numpy(float), df["vote_share"].to_numpy(float))
    return r0, r1, r1 - r0


def main() -> None:
    tf = theme_features()
    themes = list(THEMES)

    # ---------- 지역축: 테마별 개별 ΔR² (구도 통제) ----------
    reg = assemble_region().merge(tf, on=["election_id", "slot"], how="left").fillna(0)
    reg["sA"] = (reg.slot == "A").astype(float); reg["sB"] = (reg.slot == "B").astype(float)
    base_r = ["sA", "sB", "regional_base"]
    print(f"● 지역축 ({len(reg)}행) 테마별 ΔR² (슬롯+구도 통제 후)")
    rows = [(t, dr2(reg, base_r, [t])[2]) for t in themes]
    for t, d in sorted(rows, key=lambda x: -x[1]):
        print(f"    {t:6s} ΔR² = {d:+.4f}")
    print(f"    [전체 테마 동시] ΔR² = {dr2(reg, base_r, themes)[2]:+.4f}\n")

    # ---------- 세대축: 테마×세대(청년성향) 상호작용 ΔR² ----------
    gen = assemble_generation().merge(tf, on=["election_id", "slot"], how="left").fillna(0)
    gen["youth"] = gen["generation"].astype(str).map(YOUTHNESS)
    gen["sA"] = (gen.slot == "A").astype(float); gen["sB"] = (gen.slot == "B").astype(float)
    for t in themes:
        gen[t + "_x"] = gen[t] * gen["youth"]  # 테마 × 세대 매칭
    base_g = ["sA", "sB", "gen_base"]
    print(f"● 세대축 ({len(gen)}행) 테마×세대 상호작용 ΔR² (슬롯+세대구도 통제 후)")
    rows = [(t, dr2(gen, base_g, [t + "_x"])[2]) for t in themes]
    for t, d in sorted(rows, key=lambda x: -x[1]):
        print(f"    {t:6s}×세대 ΔR² = {d:+.4f}")
    print(f"    [전체 동시] ΔR² = {dr2(gen, base_g, [t + '_x' for t in themes])[2]:+.4f}")
    print("    주의: 세대 N=35 → 방향·상대크기 참고용(과적합 경계)")


if __name__ == "__main__":
    main()
