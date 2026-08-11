"""Simple issue-vote analysis for the presidential issue engine (AI-free).

이 스크립트는 단순 이슈-표심 분석 한 개를 담는다. 복잡한 파이프라인 없이,
고정 데이터셋(CSV)만 읽어 "이슈 지표가 실제 득표율을 얼마나 설명하는가"를
빈도·비율·%p 오차로 보여준다.

원칙
----
- AI / 오픈웨이트 모델 / LLM을 호출하지 않는다. 입력 CSV의 `*_count`, `score`
  값은 모두 규칙 기반(rule-based) 집계 결과로 간주한다.
- 의존성은 pandas 만 필수. 그림은 matplotlib 가 있으면 그리고, 없으면 표만
  남긴다(설치 강제 안 함).
- 채점 자(척도)는 공용 `common.evaluation` 을 그대로 쓴다 → 오픈소스 대회의
  복잡한 엔진과 같은 %p 기준으로 비교 가능하다.

산출물 (네 가지)
----------------
1. 후보(슬롯)별 이슈 노출량      : candidate_tone_scores.mention_count
2. 후보(슬롯)별 부정 키워드 비율 : negative_frame_count / mention_count
3. 여론조사 흐름                 : polls.csv 가 있으면 시계열, 없으면 생략
4. 실제 vs 예측 득표율 %p 오차   : variable_model_predictions vs
                                   presidential_results_standardized

실행
----
    python presidential_issue_engine/simple_feature_analysis.py \
        --data-dir presidential_issue_engine/fixed_dataset \
        --election-id pres_2022 \
        --out-dir presidential_issue_engine/report \
        --figures presidential_issue_engine/poster_figures

고정 데이터셋이 아직 비어 있으면, 데모로 더미 데이터를 가리키면 된다:
    --data-dir data/presidential  (단, 더미이므로 결론용이 아님)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Windows 콘솔(cp949)에서 한글 print 깨짐 방지. 저장 CSV 는 utf-8-sig 사용.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# 공용 채점 자(common)를 쓰기 위해 레포 루트를 import 경로에 추가한다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from common.evaluation import percentage_point_errors  # noqa: E402
from common.issue_store import rollup_issue_features  # noqa: E402


def _read(data_dir: Path, name: str) -> pd.DataFrame | None:
    """CSV 가 있으면 읽고, 없으면 None (해당 분석은 건너뜀)."""

    path = data_dir / f"{name}.csv"
    if not path.exists():
        print(f"  [skip] {path.name} 없음 → 이 분석은 생략")
        return None
    return pd.read_csv(path)


def issue_exposure(tone: pd.DataFrame, election_id: str) -> pd.DataFrame:
    """1) 후보(슬롯)별 이슈 노출량 = 언급량(mention_count)."""

    frame = tone.loc[tone["election_id"] == election_id]
    out = (
        frame.groupby("slot", as_index=False)["mention_count"].sum()
        .rename(columns={"mention_count": "issue_exposure"})
        .sort_values("slot")
    )
    return out


def negative_keyword_ratio(tone: pd.DataFrame, election_id: str) -> pd.DataFrame:
    """2) 후보(슬롯)별 부정 키워드 비율 = 부정 프레임 / 전체 언급."""

    frame = tone.loc[tone["election_id"] == election_id].copy()
    grouped = frame.groupby("slot", as_index=False)[["negative_frame_count", "mention_count"]].sum()
    grouped["negative_ratio"] = grouped["negative_frame_count"] / grouped["mention_count"].where(
        grouped["mention_count"] > 0
    )
    return grouped.sort_values("slot")


def poll_trend(polls: pd.DataFrame) -> pd.DataFrame:
    """3) 여론조사 흐름 = 조사일 기준 지지율 시계열.

    고정 데이터셋의 polls.csv 는 `slot`(또는 candidate) + 조사일 + 지지율을
    가진다고 가정한다. 컬럼명이 다르면 가능한 만큼만 정리한다.
    """

    frame = polls.copy()
    date_col = next((c for c in ("end_date", "published_date", "date", "start_date") if c in frame.columns), None)
    support_col = next((c for c in ("support_rate", "support", "vote_share") if c in frame.columns), None)
    group_col = "slot" if "slot" in frame.columns else ("candidate_id" if "candidate_id" in frame.columns else None)
    if date_col is None or support_col is None or group_col is None:
        print("  [skip] polls 컬럼(date/support/slot)을 찾지 못해 여론조사 흐름 생략")
        return pd.DataFrame()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    return (
        frame.dropna(subset=[date_col])
        .sort_values(date_col)[[date_col, group_col, support_col]]
        .rename(columns={date_col: "date", group_col: "series", support_col: "support_rate"})
    )


def vote_share_error(
    predictions: pd.DataFrame, actual: pd.DataFrame, election_id: str, model_name: str | None
) -> pd.DataFrame:
    """4) 실제 vs 예측 득표율 %p 오차 (공용 채점 자 사용)."""

    pred = predictions.loc[predictions["election_id"] == election_id].copy()
    if model_name and "model_name" in pred.columns:
        pred = pred.loc[pred["model_name"] == model_name]
    act = actual.loc[actual["election_id"] == election_id].copy()
    merged = percentage_point_errors(pred, act)
    cols = [c for c in ("region_id", "region_name", "slot", "predicted_vote_share", "actual_vote_share", "error_pp", "abs_error_pp") if c in merged.columns]
    return merged[cols].sort_values([c for c in ("region_id", "slot") if c in cols])


def assemble_issue_features(
    data_dir: Path, election_id: str, forecast_date: str, half_life: float = 30.0
) -> pd.DataFrame | None:
    """이슈 메모리(issue_events) + 지역 민감도 → feature 롱포맷 (issue_store rollup)."""

    issues = _read(data_dir, "issue_events")
    sens = _read(data_dir, "region_issue_sensitivity")
    if issues is None or sens is None:
        return None
    issues = issues.loc[issues["election_id"] == election_id]
    if issues.empty:
        return None
    return rollup_issue_features(issues, sens, forecast_date=forecast_date, half_life_days=half_life)


def feature_matrix(features_long: pd.DataFrame, actual: pd.DataFrame, election_id: str) -> pd.DataFrame:
    """feature 롱포맷을 와이드로 피벗 후 실제 득표율과 병합 (region×slot 단위)."""

    act = actual.loc[actual["election_id"] == election_id, ["region_id", "slot", "vote_share"]].copy()
    wide = features_long.pivot_table(
        index=["region_id", "slot"], columns="variable_name", values="variable_value", aggfunc="sum"
    ).reset_index()
    return wide.merge(act, on=["region_id", "slot"], how="inner")


def correlation_table(matrix: pd.DataFrame, var_cols: list[str]) -> pd.DataFrame:
    """5a) 각 이슈 변수와 실제 득표율의 피어슨 상관계수."""

    rows = []
    for v in var_cols:
        r = float("nan") if matrix[v].std(skipna=True) == 0 else matrix[v].corr(matrix["vote_share"])
        rows.append({"variable": v, "pearson_r": r, "n": int(matrix[v].notna().sum())})
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def ols_regression(matrix: pd.DataFrame, var_cols: list[str]) -> tuple[pd.DataFrame | None, float]:
    """5b) 득표율 ~ 이슈 변수 OLS. 계수 + R² (numpy). statsmodels 있으면 p값 추가."""

    df = matrix.dropna(subset=var_cols + ["vote_share"])
    if len(df) <= len(var_cols) + 1:
        return None, float("nan")
    X = df[var_cols].to_numpy(dtype=float)
    y = df["vote_share"].to_numpy(dtype=float)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    coef = pd.DataFrame({"term": ["const"] + list(var_cols), "coef": beta})
    try:  # 선택: 정확한 R²·p값 (포스터/발표에 좋음). 없으면 numpy R²만.
        import statsmodels.api as sm

        fit = sm.OLS(y, Xd).fit()
        coef["p_value"] = fit.pvalues
        r2 = float(fit.rsquared)
    except Exception:
        print("  [info] statsmodels 없음 → p값 생략 (R²·계수는 numpy로 산출)")
    return coef, r2


def _save(frame: pd.DataFrame, out_dir: Path, name: str) -> None:
    if frame is None or frame.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved {path}")


def _try_plot_bar(frame: pd.DataFrame, x: str, y: str, title: str, figures: Path, fname: str) -> None:
    if frame is None or frame.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"  [info] matplotlib 없음 → {fname} 그림 생략 (표는 저장됨)")
        return
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(frame[x].astype(str), frame[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    path = figures / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figure {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Presidential issue-vote simple analysis (AI-free)")
    parser.add_argument("--data-dir", default="presidential_issue_engine/fixed_dataset", type=Path)
    parser.add_argument("--election-id", default="pres_2022")
    parser.add_argument("--model-name", default=None, help="예측 모델 이름 필터 (예: balanced)")
    parser.add_argument("--forecast-date", default="2022-03-08", help="이슈 누수 차단 시점")
    parser.add_argument("--out-dir", default="presidential_issue_engine/report/tables", type=Path)
    parser.add_argument("--figures", default="presidential_issue_engine/poster_figures", type=Path)
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    print(f"[전국구] 데이터: {data_dir}  선거: {args.election_id}")
    if not data_dir.exists():
        print(f"  데이터 폴더가 없습니다: {data_dir}")
        print("  고정 데이터셋을 채우거나 데모로 --data-dir data/presidential 를 쓰세요.")
        return 1

    tone = _read(data_dir, "candidate_tone_scores")
    preds = _read(data_dir, "variable_model_predictions")
    actual = _read(data_dir, "presidential_results_standardized")
    polls = _read(data_dir, "polls")

    if tone is not None:
        print("1) 후보(슬롯)별 이슈 노출량")
        exposure = issue_exposure(tone, args.election_id)
        print(exposure.to_string(index=False))
        _save(exposure, args.out_dir, "issue_exposure")
        _try_plot_bar(exposure, "slot", "issue_exposure", "이슈 노출량 (슬롯별)", args.figures, "issue_exposure.png")

        print("2) 후보(슬롯)별 부정 키워드 비율")
        neg = negative_keyword_ratio(tone, args.election_id)
        print(neg[["slot", "negative_ratio"]].to_string(index=False))
        _save(neg, args.out_dir, "negative_keyword_ratio")
        _try_plot_bar(neg, "slot", "negative_ratio", "부정 키워드 비율 (슬롯별)", args.figures, "negative_ratio.png")

    if polls is not None:
        print("3) 여론조사 흐름")
        trend = poll_trend(polls)
        if not trend.empty:
            print(trend.head(10).to_string(index=False))
            _save(trend, args.out_dir, "poll_trend")

    if preds is not None and actual is not None:
        print("4) 실제 vs 예측 득표율 %p 오차")
        err = vote_share_error(preds, actual, args.election_id, args.model_name)
        if err.empty:
            print("  병합 결과 없음 (election_id / model_name 확인)")
        else:
            print(err.to_string(index=False))
            print(f"  ▶ 평균 절대 오차(MAE): {err['abs_error_pp'].mean():.2f} %p")
            _save(err, args.out_dir, "vote_share_error_pp")

    feats = assemble_issue_features(data_dir, args.election_id, args.forecast_date)
    if feats is not None and not feats.empty and actual is not None:
        print("5) 이슈변수 ↔ 득표율: 상관 + 회귀 (설명력)")
        matrix = feature_matrix(feats, actual, args.election_id)
        var_cols = [c for c in feats["variable_name"].unique() if c in matrix.columns]
        if matrix.empty or not var_cols:
            print("  병합/변수 없음 (region_id 정합 확인 — 이슈 지역범위 vs 결과 region_id)")
        else:
            corr = correlation_table(matrix, var_cols)
            print(corr.to_string(index=False))
            _save(corr, args.out_dir, "issue_vote_correlation")
            coef, r2 = ols_regression(matrix, var_cols)
            if coef is None:
                print(f"  표본(n={len(matrix)})이 변수 수보다 적어 회귀 생략")
            else:
                print(f"  ▶ 회귀 R² = {r2:.3f}  (이슈 변수가 득표율 분산의 {r2*100:.1f}% 설명)")
                print(coef.to_string(index=False))
                _save(coef, args.out_dir, "issue_vote_regression")

    print("[전국구] 분석 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
