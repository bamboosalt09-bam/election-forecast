"""Generate current V27 visualizations from finalized artifacts only.

The map uses a hash-pinned, SGIS-derived 2025-04-01 administrative snapshot. See
``docs/VISUALIZATION_DATA.md`` for provenance and reproduction details.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Wedge
    from shapely.geometry import shape
    from shapely.ops import unary_union
except ModuleNotFoundError as exc:
    VIZ_IMPORT_ERROR = exc
else:
    VIZ_IMPORT_ERROR = None

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "poster_figures"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v27"
FORECAST_DIR = ROOT / "outputs" / "prospective_pres_2025_v27"
REGIONS = Path(__file__).resolve().parent / "fixed_dataset" / "regions_master.csv"
MAP_URL = (
    "https://raw.githubusercontent.com/vuski/admdongkor/"
    "fbcac3db020609dce5831a856a6d5aa5cb40a908/"
    "ver20250401/HangJeongDong_ver20250401.geojson"
)
MAP_SHA256 = "1b80c423c82a9349859aef020174c1276896943d064762fc2e184f75f5ee2ceb"

NAVY, BLUE, RED, ORANGE, GRAY = "#17365D", "#0878D1", "#E33D3D", "#F28C28", "#7B8794"
SLOT_COLORS = {"A": BLUE, "B": RED, "C": ORANGE}
SOURCE_SIDO_TO_REGION = {
    "11":"sido_11", "26":"sido_26", "27":"sido_27", "28":"sido_28",
    "29":"sido_29", "30":"sido_30", "31":"sido_31", "36":"sido_36",
    "41":"sido_41", "51":"sido_42", "43":"sido_43", "44":"sido_44",
    "52":"sido_45", "46":"sido_46", "47":"sido_47", "48":"sido_48",
    "50":"sido_50",
}
PIE_ANCHORS = {
    "sido_11":(126.98,37.57), "sido_28":(126.63,37.46), "sido_41":(127.25,37.25),
    "sido_42":(128.20,37.40), "sido_44":(126.80,36.55), "sido_36":(127.29,36.48),
    "sido_30":(127.38,36.35), "sido_43":(127.70,36.80), "sido_45":(127.15,35.75),
    "sido_29":(126.85,35.16), "sido_46":(126.65,34.75), "sido_47":(128.75,36.45),
    "sido_27":(128.60,35.87), "sido_48":(128.25,35.25), "sido_31":(129.31,35.54),
    "sido_26":(129.08,35.18), "sido_50":(126.55,33.38),
}
SHORT_REGION_NAMES = {
    "sido_11":"서울", "sido_26":"부산", "sido_27":"대구", "sido_28":"인천",
    "sido_29":"광주", "sido_30":"대전", "sido_31":"울산", "sido_36":"세종",
    "sido_41":"경기", "sido_42":"강원", "sido_43":"충북", "sido_44":"충남",
    "sido_45":"전북", "sido_46":"전남", "sido_47":"경북", "sido_48":"경남",
    "sido_50":"제주",
}


def _require_viz() -> None:
    if VIZ_IMPORT_ERROR is not None:
        raise SystemExit('Install visualization dependencies with: pip install -e ".[viz]"') from VIZ_IMPORT_ERROR


def setup_style() -> None:
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in installed), "DejaVu Sans")
    if selected == "DejaVu Sans":
        print("WARNING: Korean font unavailable; labels may render incorrectly.", file=sys.stderr)
    plt.rcParams.update({"font.family":[selected,"DejaVu Sans"], "axes.unicode_minus":False})


def _regions() -> pd.DataFrame:
    return pd.read_csv(REGIONS, encoding="utf-8-sig")[["region_id","region_name"]]


def _history() -> pd.DataFrame:
    frame = pd.read_csv(ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    required = {"election_id","region_id","slot","candidate_name","actual","layer_pred"}
    if missing := required - set(frame):
        raise ValueError(f"V27 history missing columns: {sorted(missing)}")
    if frame["election_id"].astype(str).str.contains("2025").any():
        raise ValueError("2025 leaked into retrospective visualization")
    return frame.merge(_regions(), on="region_id", how="left", validate="many_to_one")


def _forecast() -> pd.DataFrame:
    frame = pd.read_csv(FORECAST_DIR / "prospective_predictions.csv", encoding="utf-8-sig")
    required = {"election_id","region_id","slot","candidate_name","predicted_share"}
    if missing := required - set(frame):
        raise ValueError(f"V27 forecast missing columns: {sorted(missing)}")
    if set(frame["election_id"].astype(str)) != {"pres_2025"}:
        raise ValueError("forecast visualization accepts only pres_2025")
    totals = frame.groupby("region_id")["predicted_share"].sum()
    if len(totals) != 17 or not ((totals - 1).abs() < 1e-9).all():
        raise ValueError("forecast must have 17 compositional regions")
    return frame.merge(_regions(), on="region_id", how="left", validate="many_to_one")


def _korea_shapes() -> list[tuple[str,list[list[tuple[float,float]]]]]:
    with urlopen(MAP_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != MAP_SHA256:
        raise ValueError(f"administrative-boundary hash mismatch: {digest}")
    source = json.loads(payload.decode("utf-8-sig"))
    grouped = {}
    for feature in source["features"]:
        code = str(feature["properties"]["sido"])
        grouped.setdefault(code, []).append(shape(feature["geometry"]))
    output = []
    for code, geometries in grouped.items():
        region = SOURCE_SIDO_TO_REGION.get(code)
        if region is None:
            continue
        merged = unary_union(geometries)
        polygons = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        output.append((region, [[tuple(point) for point in polygon.exterior.coords] for polygon in polygons]))
    if {region for region,_ in output} != set(SOURCE_SIDO_TO_REGION.values()):
        raise ValueError("SGIS-derived input does not resolve all 17 regions")
    return output


def _bounds(parts):
    points = [point for part in parts for point in part]
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_performance():
    metrics = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))["metrics"]
    values = [metrics["regional_equal_election_macro_mae_pp"], metrics["national_equal_election_macro_mae_pp"]]
    fig, ax = plt.subplots(figsize=(8.6,5.2))
    bars = ax.bar(["지역 MAE","전국 MAE"], values, color=[BLUE,RED], width=.58)
    ax.set_ylabel("동일 선거 가중 MAE (%p)")
    ax.set_title("V27 회고 개발 패널 성능", fontsize=18, fontweight="bold", color=NAVY)
    ax.grid(axis="y", alpha=.2)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.07, f"{value:.3f}%p", ha="center", fontweight="bold")
    ax.text(.5,-.17,"2002–2022 개발 패널 · 독립 미래 검증 아님 · 2025 결과 미포함",transform=ax.transAxes,ha="center",color=GRAY)
    fig.tight_layout()
    return fig


def performance_by_election():
    frame = pd.read_csv(ACTIVE_DIR / "by_election.csv", encoding="utf-8-sig")
    frame["year"] = frame["election_id"].str.replace("pres_","",regex=False)
    fig, ax = plt.subplots(figsize=(10.2,5.4))
    bars = ax.bar(frame["year"], frame["regional_weighted_mae_pp"], color=BLUE)
    ax.set_ylabel("지역 MAE (%p)")
    ax.set_title("V27 선거별 지역 오차", fontsize=18, fontweight="bold", color=NAVY)
    ax.grid(axis="y", alpha=.2)
    for bar,value in zip(bars,frame["regional_weighted_mae_pp"]):
        ax.text(bar.get_x()+bar.get_width()/2,value+.08,f"{value:.2f}",ha="center")
    fig.tight_layout()
    return fig


def regional_pred_vs_actual(election_id: str):
    election = _history().loc[lambda x: x["election_id"].eq(election_id)]
    if election.empty:
        raise ValueError(f"no V27 rows for {election_id}")
    slots = sorted(election["slot"].astype(str).unique())
    fig, axes = plt.subplots(1,len(slots),figsize=(5.7*len(slots),8.2),sharex=True,squeeze=False)
    for axis,slot in zip(axes[0],slots):
        rows = election.loc[election["slot"].astype(str).eq(slot)].sort_values("actual")
        y = list(range(len(rows)))
        axis.barh([v-.18 for v in y],rows["layer_pred"]*100,height=.34,color=BLUE,label="예측")
        axis.barh([v+.18 for v in y],rows["actual"]*100,height=.34,color=GRAY,label="실제")
        axis.set_yticks(y,rows["region_name"]); axis.set_xlim(0,100); axis.grid(axis="x",alpha=.2)
        axis.set_title(f"{slot} · {rows['candidate_name'].iloc[0]}",color=NAVY,fontweight="bold")
        axis.set_xlabel("득표율 (%)")
    axes[0][0].legend(loc="lower right")
    fig.suptitle(f"{election_id[-4:]} 대선 V27 지역 예측과 실제",fontsize=18,fontweight="bold",color=NAVY)
    fig.tight_layout(); return fig


def prospective_bars():
    frame = _forecast(); slots = sorted(frame["slot"].astype(str).unique())
    fig, axes = plt.subplots(1,len(slots),figsize=(5.7*len(slots),8.2),sharex=True,squeeze=False)
    for axis,slot in zip(axes[0],slots):
        rows = frame.loc[frame["slot"].astype(str).eq(slot)].sort_values("predicted_share")
        axis.barh(rows["region_name"],rows["predicted_share"]*100,color=SLOT_COLORS[slot])
        axis.set_xlim(0,100); axis.grid(axis="x",alpha=.2)
        axis.set_title(f"{slot} · {rows['candidate_name'].iloc[0]}",color=NAVY,fontweight="bold")
        axis.set_xlabel("예측 득표율 (%)")
    fig.suptitle("2025 대선 V27 D-1 지역 예측\n실제 결과 미사용 · 성능평가 대상 아님",fontsize=18,fontweight="bold",color=NAVY)
    fig.tight_layout(); return fig


def prospective_map():
    frame, shapes = _forecast(), _korea_shapes()
    fig = plt.figure(figsize=(14.5,9.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=.02)
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])
    all_parts = [p for _,parts in shapes for p in parts]
    for region,parts in shapes:
        for part in parts:
            ax.add_patch(Polygon(part,closed=True,facecolor="#F4F7FA",edgecolor="#9AA6B2",linewidth=.65,zorder=1))
        x,y = PIE_ANCHORS[region]
        rows=frame.loc[frame["region_id"].eq(region)].sort_values("slot"); start=90.
        for row in rows.itertuples(index=False):
            sweep=float(row.predicted_share)*360
            ax.add_patch(Wedge((x,y),.067,start-sweep,start,facecolor=SLOT_COLORS[str(row.slot)],edgecolor="white",linewidth=.4,zorder=3)); start-=sweep
        ax.text(x,y-.083,SHORT_REGION_NAMES[region],ha="center",va="top",fontsize=6.2,color="#263442",zorder=4)
    left,bottom,right,top=_bounds(all_parts); ax.set_xlim(left-.35,right+.35); ax.set_ylim(bottom-.2,top+.2)
    ax.set_aspect("equal"); ax.axis("off")
    candidates=frame[["slot","candidate_name"]].drop_duplicates().sort_values("slot")
    handles=[plt.Line2D([0],[0],marker="o",color="none",markerfacecolor=SLOT_COLORS[r.slot],markersize=11,label=f"{r.slot} · {r.candidate_name}") for r in candidates.itertuples(index=False)]
    ax.legend(handles=handles,loc="lower left",frameon=False,fontsize=9)
    ax.set_title("지역별 예측 구성",fontsize=17,fontweight="bold",color=NAVY,pad=10)

    wide = frame.pivot(index=["region_id","region_name"],columns="slot",values="predicted_share").reset_index()
    order = list(_regions()["region_id"])
    wide["order"] = wide["region_id"].map({value:index for index,value in enumerate(order)})
    wide = wide.sort_values("order")
    cell_text = [[row.region_name, f"{row.A*100:.1f}", f"{row.B*100:.1f}", f"{row.C*100:.1f}"] for row in wide.itertuples(index=False)]
    table = table_ax.table(cellText=cell_text, colLabels=["시·도","A (%)","B (%)","C (%)"], cellLoc="center", colLoc="center", loc="center", colWidths=[.40,.20,.20,.20])
    table.auto_set_font_size(False); table.set_fontsize(8.3); table.scale(1,1.25)
    for column,color in enumerate((NAVY,BLUE,RED,ORANGE)):
        table[(0,column)].set_facecolor(color); table[(0,column)].set_text_props(color="white",weight="bold")
    for (row,column),cell in table.get_celld().items():
        cell.set_edgecolor("#D6DDE4"); cell.set_linewidth(.55)
        if row and row % 2 == 0: cell.set_facecolor("#F7F9FB")
    table_ax.axis("off"); table_ax.set_title("시·도별 예측 득표율",fontsize=17,fontweight="bold",color=NAVY,pad=10)
    fig.suptitle("2025 대선 V27 D-1 예측 지도",fontsize=21,fontweight="bold",color=NAVY,y=.975)
    fig.text(.5,.025,"각 원은 해당 시·도의 예측 득표 구성(합계 100%) · 원 크기는 인구를 뜻하지 않음\n예측 기준 2025-06-02 (D-1) · 실제 결과 미사용 · 경계: 통계청 SGIS 기반 admdongkor 2025-04-01 (CC BY 4.0 / 공공누리 제1유형)",ha="center",va="bottom",fontsize=9.0,color=GRAY)
    fig.subplots_adjust(left=.03,right=.98,bottom=.11,top=.88,wspace=.02)
    return fig


def main() -> None:
    _require_viz(); setup_style()
    figures={"v27_model_performance":model_performance(),"v27_performance_by_election":performance_by_election(),
        **{f"v27_regional_{e}":regional_pred_vs_actual(e) for e in ("pres_2002","pres_2007","pres_2012","pres_2017","pres_2022")},
        "v27_pres_2025_regional_bars":prospective_bars(),"v27_pres_2025_regional_map":prospective_map()}
    for name,figure in figures.items(): _save(figure,name)
    print(f"Generated {len(figures)} current V27 PNG files in {OUT}")


if __name__ == "__main__":
    main()
