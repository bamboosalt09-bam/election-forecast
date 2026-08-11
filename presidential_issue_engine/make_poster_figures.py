"""Generate final statistics-competition poster figures.

The figures intentionally use the frozen values reproduced by
``presidential_issue_engine/robustness_check.py``.  They are presentation artifacts, not
another modeling step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "poster_figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1f3b6f"
BLUE = "#2e86de"
RED = "#c0392b"
GREEN = "#1e8449"
GRAY = "#7f8c8d"
LIGHT_BLUE = "#eef3fb"
LIGHT_GREEN = "#e8f6ef"
LIGHT_RED = "#fdecea"

MAE_LADDER = [
    ("전체평균", 19.93),
    ("슬롯평균", 15.71),
    ("슬롯+지역구도", 8.67),
    ("슬롯+지역+이슈", 8.42),
    ("NEC prior+이슈 Ridge", 6.90),
]

ROLLING_BY_ELECTION = [
    ("2007", 5.95),
    ("2012", 7.22),
    ("2017", 7.80),
    ("2022", 7.75),
]

R2_LADDER = [
    ("슬롯만", 0.304),
    ("슬롯+지역구도", 0.799),
    ("슬롯+지역+이슈", 0.823),
    ("NEC prior+이슈 Ridge", 0.877),
]

REGION_ROLLING = [
    ("강원", 9.40),
    ("PK", 9.73),
    ("전체", 10.44),
    ("TK", 10.67),
    ("호남", 11.55),
]

VARIABLE_IMPORTANCE = [
    ("slot_A", 0.582),
    ("slot_B", 0.433),
    ("slotB_prior", 0.361),
    ("rif", 0.359),
    ("slotA_prior", 0.313),
    ("landscape_bloc_alignment", -0.312),
    ("landscape_centrist", 0.284),
    ("partisan_prior", 0.259),
    ("issue_advantage", -0.180),
    ("landscape_inferred_prior", 0.042),
]


def setup_style() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def mae_ladder() -> plt.Figure:
    labels = [item[0] for item in MAE_LADDER]
    values = [item[1] for item in MAE_LADDER]
    colors = [GRAY, GRAY, BLUE, BLUE, RED]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xlabel("LOEO MAE (%p, 낮을수록 좋음)")
    ax.set_title("모델 단계별 MAE 감소", fontsize=18, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%p", va="center", fontsize=12)
    ax.text(
        0.01,
        -0.16,
        "핵심 흐름: 전체평균 → 후보 슬롯 → 지역구도 → 이슈 → Ridge 보정",
        transform=ax.transAxes,
        fontsize=11,
        color=GRAY,
    )
    fig.tight_layout()
    return fig


def r2_ladder() -> plt.Figure:
    labels = [item[0] for item in R2_LADDER]
    values = [item[1] for item in R2_LADDER]
    colors = [GRAY, BLUE, BLUE, RED]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("R²")
    ax.set_title("설명력(R²) 사다리", fontsize=18, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=10)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=12)
    fig.tight_layout()
    return fig


def region_error() -> plt.Figure:
    labels = [item[0] for item in REGION_ROLLING]
    values = [item[1] for item in REGION_ROLLING]
    colors = [BLUE, BLUE, NAVY, BLUE, RED]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 13)
    ax.set_ylabel("Rolling-origin MAE (%p)")
    ax.set_title("지역별 rolling-origin 오차", fontsize=18, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}", ha="center", fontsize=12)
    ax.text(
        0.01,
        -0.16,
        "호남은 2017년 제3후보 분열처럼 구조 prior만으로 잡기 어려운 사건성이 남는다.",
        transform=ax.transAxes,
        fontsize=11,
        color=GRAY,
    )
    fig.tight_layout()
    return fig


def variable_importance() -> plt.Figure:
    data = sorted(VARIABLE_IMPORTANCE, key=lambda item: item[1])
    labels = [item[0] for item in data]
    values = [item[1] for item in data]
    colors = [RED if value < 0 else (GREEN if label == "rif" else BLUE) for label, value in data]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("표준화 β")
    ax.set_title("핵심 변수 중요도", fontsize=18, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        offset = 0.018 if value >= 0 else -0.018
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha=ha, fontsize=10)
    ax.text(
        0.01,
        -0.12,
        "후보 슬롯과 지역 prior가 기본 구조를 만들고, rif가 이슈의 추가 설명력을 담당한다.",
        transform=ax.transAxes,
        fontsize=11,
        color=GRAY,
    )
    fig.tight_layout()
    return fig


def research_flow() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.axis("off")
    ax.set_title("연구 흐름 요약도", fontsize=18, fontweight="bold", color=NAVY, pad=12)

    boxes = [
        ("국회 발언록", "이슈 키워드 매칭\n발언량·정당별 강조도", LIGHT_BLUE),
        ("지역구도 prior", "NEC 대선·비례·지방의회\n지역별 진영 기반", "#fff4e6"),
        ("이슈-지역 결합", "후보 이슈소유 × 이슈부각\n× 지역 민감도", LIGHT_GREEN),
        ("Ridge 모델", "과적합 억제\n최종 LOEO 6.90%p", LIGHT_RED),
        ("검증", "rolling-origin 7.14%p\nR² 0.877", "#f4ecf7"),
    ]

    x = 0.03
    y = 0.39
    width = 0.16
    height = 0.30
    for index, (title, body, color) in enumerate(boxes):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.018",
                fc=color,
                ec=NAVY,
                lw=1.4,
                transform=ax.transAxes,
            )
        )
        ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY, transform=ax.transAxes)
        ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center", fontsize=10, color="#222222", transform=ax.transAxes)
        if index < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.012, y + height / 2),
                    (x + width + 0.055, y + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    color=GRAY,
                    transform=ax.transAxes,
                )
            )
        x += 0.195

    ax.text(
        0.5,
        0.16,
        "결론: 대선 득표율은 후보 슬롯과 지역구도가 기본이고, 국회 발언록 기반 이슈 지표는 그 잔차를 추가로 줄인다.",
        ha="center",
        va="center",
        fontsize=12,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=GRAY, lw=1.0),
        transform=ax.transAxes,
    )
    return fig


def rolling_by_election() -> plt.Figure:
    labels = [item[0] for item in ROLLING_BY_ELECTION]
    values = [item[1] for item in ROLLING_BY_ELECTION]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bars = ax.bar(labels, values, color=[BLUE, BLUE, RED, RED])
    ax.axhline(7.14, color=NAVY, linestyle="--", linewidth=1.6)
    ax.text(3.05, 7.25, "평균 7.14%p", color=NAVY, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 9)
    ax.set_ylabel("Rolling-origin MAE (%p)")
    ax.set_title("선거별 rolling-origin 오차", fontsize=18, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}", ha="center", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    figures = {
        "01_model_mae_ladder": mae_ladder(),
        "02_r2_ladder": r2_ladder(),
        "03_region_rolling_error": region_error(),
        "04_variable_importance": variable_importance(),
        "05_research_flow": research_flow(),
        "06_rolling_by_election": rolling_by_election(),
    }
    for name, fig in figures.items():
        save(fig, name)
    print(f"Generated {len(figures)} PNG files in {OUT}")


if __name__ == "__main__":
    main()
