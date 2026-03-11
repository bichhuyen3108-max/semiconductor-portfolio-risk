import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
from src.risk_metrics import run_var_backtest_kupiec
from src.portfolio import make_equal_weights, portfolio_return
from matplotlib.ticker import MultipleLocator

# 한글 폰트 설정
mpl.rcParams["font.family"] = "Malgun Gothic"

# 마이너스 깨짐 방지
mpl.rcParams["axes.unicode_minus"] = False

def plot_var_cvar_multi_levels(
    csv_path: str = "data/processed/log_returns.csv",
    conf_levels=(0.95, 0.99),
    save_path: str = "results/figures/var_cvar_multi_levels.png",
):
    """
    여러 신뢰수준(conf_levels)에 대해 Historical VaR/CVaR를 동시에 시각화
    - 동일가중(Equal-weight) 포트폴리오 수익률 기준
    """

    # ===============================
    # 1) 로그수익률 데이터 로드
    # ===============================
    df = pd.read_csv(csv_path)
    returns = df.drop(columns=["Date"], errors="ignore").dropna().astype(float)

    # ===============================
    # 2) 동일가중 포트폴리오 수익률
    # ===============================
    w = make_equal_weights(returns.columns)
    port = portfolio_return(returns, w)   # Series name="portfolio"

    # ===============================
    # 3) 분포 히스토그램(전체) 먼저 그림
    # ===============================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(port, bins=60, density=True, label="Portfolio Returns")

    # ===============================
    # 4) 각 신뢰수준별 VaR/CVaR 계산 + 표시
    # ===============================
    for alpha_conf in conf_levels:
        q = 1 - alpha_conf  # left-tail 확률 (예: 0.95 -> 0.05)

        var = port.quantile(q)
        tail = port[port <= var]
        cvar = float(tail.mean())

        print(f"[INFO] conf={alpha_conf:.2f}, q={q:.2f}")
        print(f"       VaR  = {var:.6f} (loss {-var:.2%})")
        print(f"       CVaR = {cvar:.6f} (loss {-cvar:.2%})")
        print(f"       tail count: {len(tail)}/{len(port)}")

        # tail 히스토그램(옵션): 너무 겹치면 주석처리 가능
        ax.hist(tail, bins=30, density=True, alpha=0.6, label=f"Tail <= VaR ({q:.0%})")

        # VaR / CVaR 기준선
        ax.axvline(var, linestyle="--", linewidth=2, label=f"VaR {alpha_conf:.0%} = {var:.4f}")
        ax.axvline(cvar, linestyle=":", linewidth=2, label=f"CVaR {alpha_conf:.0%} = {cvar:.4f}")

    ax.set_title("Portfolio Return Distribution with Historical VaR/CVaR (Multi Levels)")
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()

    # ===============================
    # 5) 저장
    # ===============================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def load_portfolio_and_rolling_var(
    log_returns_path: str = "data/processed/log_returns.csv",
    rolling_var_path: str = "results/tables/rolling_var.csv",
):
    """
    (KR) 로그수익률 CSV와 Rolling VaR CSV를 Date 기준으로 정렬/결합하여
         포트폴리오 수익률 + 이동 VaR 데이터프레임 생성
    """
    # 1) 로그수익률 로드
    log_df = pd.read_csv(log_returns_path)
    if "Date" not in log_df.columns:
        raise ValueError("log_returns.csv 파일에 'Date' 컬럼이 없습니다.")

    # Date 파싱
    log_df["Date"] = pd.to_datetime(log_df["Date"])

    # 종목 수익률만 추출
    returns = log_df.drop(columns=["Date"], errors="ignore").dropna().astype(float)

    w = make_equal_weights(returns.columns)
    port = portfolio_return(returns, w)   # index aligned with returns.index


    # 포트폴리오 수익률 DF
    port_df = pd.DataFrame({
        "Date": log_df.loc[returns.index, "Date"].values,
        "portfolio": port.values
    })

    # 2) Rolling VaR 로드
    rolling_df = pd.read_csv(rolling_var_path)
    if "Date" not in rolling_df.columns:
        raise ValueError("rolling_var.csv 파일에 'Date' 컬럼이 없습니다.")

    rolling_df["Date"] = pd.to_datetime(rolling_df["Date"])

    # 3) Date 기준으로 결합 (inner join: 공통 날짜만)
   # ✅ nếu rolling_df lỡ có cột portfolio cũ thì bỏ nó trước khi merge
    rolling_df = rolling_df.drop(columns=["portfolio", "Portfolio_Return"], errors="ignore")

    merged = (
        pd.merge(rolling_df, port_df, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # ✅ đảm bảo chỉ 1 cột portfolio
    if "portfolio_x" in merged.columns and "portfolio_y" in merged.columns:
        merged = merged.drop(columns=["portfolio_x"]).rename(columns={"portfolio_y": "portfolio"})
    elif "portfolio_y" in merged.columns:
        merged = merged.rename(columns={"portfolio_y": "portfolio"})
    elif "portfolio_x" in merged.columns:
        merged = merged.rename(columns={"portfolio_x": "portfolio"})

    # 필수 컬럼 체크
    for col in ["Rolling_VaR_95", "Rolling_VaR_99"]:
        if col not in merged.columns:
            raise ValueError(f"rolling_var.csv 파일에 '{col}' 컬럼이 없습니다.")

    return port, merged


def add_violations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "portfolio" in out.columns:
        ret_col = "portfolio"
    elif "Portfolio_Return" in out.columns:
        ret_col = "Portfolio_Return"
    else:
        raise KeyError(f"[add_violations] return column not found. columns={list(out.columns)}")

    out["Violation_95"] = out[ret_col] < out["Rolling_VaR_95"]
    out["Violation_99"] = out[ret_col] < out["Rolling_VaR_99"]
    return out

def plot_return_vs_rolling_var(
    df: pd.DataFrame,
    level: int = 95,
    save_path: str | None = "results/figures/rolling_var_violation_95.png",
    show: bool = True,
):
    """
    (KR) 포트폴리오 수익률 시계열 + 이동 VaR 오버레이 + 위반(빨간 점) 표시
    

    level: 95 또는 99
    """
    if level not in (95, 99):
        raise ValueError("level은 95 또는 99만 가능합니다.")

    var_col = f"Rolling_VaR_{level}"
    vio_col = f"Violation_{level}"

    if var_col not in df.columns or vio_col not in df.columns:
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {var_col}, {vio_col}")

    # violation rate 출력 (backtesting의 핵심 지표 중 하나)
    violation_rate = df[vio_col].mean()
    expected_rate = 0.05 if level == 95 else 0.01
    print(f"[INFO] VaR {level}% 위반율(실제) = {violation_rate:.4f}, 기대값 = {expected_rate:.4f}")
    print(f"[INFO] 위반 횟수 = {df[vio_col].sum()} / {len(df)}")

    plt.figure(figsize=(14, 6))

    # 실제 포트폴리오 수익률
    plt.plot(df["Date"], df["portfolio"], label="Portfolio Return", alpha=0.6)

    # Rolling VaR
    plt.plot(df["Date"], df[var_col], label=f"Rolling VaR {level}%", linestyle="--")

    # 위반 포인트 강조
    vio_points = df[df[vio_col]]
    plt.scatter(
        vio_points["Date"],
        vio_points["portfolio"],
        label=f"Violation ({level}%)",
        s=18
    )

    # Y축을 더 촘촘하게 + % 표시
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.02))   # 2% 단위
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))  # 0.5% 단위
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.grid(True, which="minor", axis="y", alpha=0.15)

    plt.title(f"Portfolio Return vs Rolling VaR ({level}%) with Violations")
    plt.xlabel("Date")
    plt.ylabel("Daily Log Return")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # 저장
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def calc_yearly_violation_rate_from_rolling_var(
    df: pd.DataFrame,
    level: int = 95,
    date_col: str = "Date",
    ret_col: str = "portfolio",
) -> pd.DataFrame:
    """
    (KR) Rolling VaR 결과를 이용해 연도별 Violation Rate(위반율)을 계산합니다.

    ✅ 필요한 컬럼:
    - Date (날짜)
    - portfolio (일간 수익률)
    - Rolling_VaR_{level} (예: Rolling_VaR_95 / Rolling_VaR_99)

    ✅ Violation 정의:
    - 일간 수익률 < Rolling VaR  => Violation = 1
    - 그 외 => 0
    """
    if level not in (95, 99):
        raise ValueError("level은 95 또는 99만 가능합니다.")

    var_col = f"Rolling_VaR_{level}"
    if var_col not in df.columns:
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {var_col}")
    if ret_col not in df.columns:
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {ret_col}")
    if date_col not in df.columns:
        raise ValueError(f"df에 필요한 컬럼이 없습니다: {date_col}")

    out = df.copy()

    # 날짜 형식 변환
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, ret_col, var_col])

    # (KR) Violation 컬럼 생성: 수익률이 VaR보다 더 낮으면 1
    vio_col = f"Violation_{level}"
    out[vio_col] = (out[ret_col] < out[var_col]).astype(int)

    # 연도 추출
    out["Year"] = out[date_col].dt.year

    # 연도별 위반 횟수 / 총 관측치 / 위반율 계산
    yearly = (
        out.groupby("Year")[vio_col]
           .agg(violations="sum", total_days="count")
           .reset_index()
    )
    yearly["violation_rate"] = yearly["violations"] / yearly["total_days"]

    # 이론적 기대 위반율 (95%->5%, 99%->1%)
    expected_rate = 0.05 if level == 95 else 0.01
    yearly["expected_rate"] = expected_rate
    yearly["violation_rate_pct"] = yearly["violation_rate"] * 100

    return yearly        


def run_rolling_var_plots():
    """
    (KR) 전체 실행: 데이터 로드 -> 위반 생성 -> 95%, 99% 그래프 저장/표시
    """
    _, df = load_portfolio_and_rolling_var()
    print("[DEBUG] df.columns =", df.columns.tolist())
    df = add_violations(df)

    # ===============================
    # (KR) 연도별 위반율(Yearly Violation Rate) 계산 + 출력/저장
    # ===============================
    yearly_95 = calc_yearly_violation_rate_from_rolling_var(df, level=95)
    yearly_99 = calc_yearly_violation_rate_from_rolling_var(df, level=99)

    print("\n[INFO] 연도별 VaR 95% 위반율")
    print(yearly_95.to_string(index=False))

    print("\n[INFO] 연도별 VaR 99% 위반율")
    print(yearly_99.to_string(index=False))

    # (KR) CSV로 저장 (results/tables 폴더에 저장)
    os.makedirs("results/tables", exist_ok=True)
    yearly_95.to_csv("results/tables/yearly_violation_rate_95.csv", index=False, encoding="utf-8-sig")
    yearly_99.to_csv("results/tables/yearly_violation_rate_99.csv", index=False, encoding="utf-8-sig")
    print("[INFO] Saved yearly violation rate tables to results/tables/")

    # ===============================
    # (KR) 그래프 출력/저장
    # ===============================
    plot_return_vs_rolling_var(
        df,
        level=95,
        save_path="results/figures/rolling_var_violation_95.png",
        show=True,
    )

    plot_return_vs_rolling_var(
        df,
        level=99,
        save_path="results/figures/rolling_var_violation_99.png",
        show=True,
    )


def run_backtesting_post3():
    _, df = load_portfolio_and_rolling_var()
    df = add_violations(df)

    res95 = run_var_backtest_kupiec(df, level=95)
    res99 = run_var_backtest_kupiec(df, level=99)

    print("\n[POST3] Kupiec POF Test 결과")
    for r in (res95, res99):
        print(
            f"- VaR {r['level']}% | expected={r['expected_rate']:.2%}, "
            f"actual={r['violation_rate']:.2%} ({r['violations']}/{r['n']}), "
            f"p-value={r['p_value']:.4f} => {r['verdict']}"
        )


def plot_return_vs_var_zoom(df_hist, df_models, level=95):
    """
    Return vs VaR (zoom view)
    """

    plt.figure(figsize=(15,6))

    plt.plot(
        df_models["Date"],
        df_models["portfolio"],
        label="Portfolio Return",
        alpha=0.3,
        color="gray"
    )

    plt.plot(
        df_hist["Date"],
        df_hist[f"Rolling_VaR_{level}"],
        label="Historical VaR",
        color="blue"
    )

    plt.plot(
        df_models["Date"],
        df_models[f"Rolling_tVaR_{level}"],
        label="Student-t VaR",
        color="orange"
    )

    plt.plot(
        df_models["Date"],
        df_models[f"Rolling_GARCHVaR_{level}"],
        label="GARCH VaR",
        color="green"
    )

   
    plt.ylim(-0.12, 0.05)

    plt.title(f"Portfolio Return vs VaR ({level}%)")
    plt.xlabel("Date")
    plt.ylabel("Return")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"results/figures/return_vs_var_zoom_{level}.png", dpi=300)

    plt.show()


def plot_var_models_only(df_hist, df_models, level=95):
    """
    VaR model comparison (Historical vs Student-t vs GARCH)
    """

    plt.figure(figsize=(15,6))

    plt.plot(
        df_hist["Date"],
        df_hist[f"Rolling_VaR_{level}"],
        label="Historical VaR",
        color="blue"
    )

    plt.plot(
        df_models["Date"],
        df_models[f"Rolling_tVaR_{level}"],
        label="Student-t VaR",
        color="orange"
    )

    plt.plot(
        df_models["Date"],
        df_models[f"Rolling_GARCHVaR_{level}"],
        label="GARCH VaR",
        color="green"
    )

    plt.title(f"VaR Model Comparison ({level}%)")
    plt.xlabel("Date")
    plt.ylabel("VaR")

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"results/figures/var_models_only_{level}.png", dpi=300)

    plt.show()  

def plot_forecast_var_risk_threshold_signal(
    df: pd.DataFrame,
    date_col: str = "Date",
    var_col: str = "VaR_return_h",
    exposure_col: str = "suggested_exposure",
    reduce_signal_col: str = "reduce_signal",
    moderate_threshold_col: str = "moderate_threshold",
    high_threshold_col: str = "high_threshold",
    save_path: str = "results/figures/risk_threshold_signal.png",
    show: bool = True,
):
    """
    Forecast VaR 기반 threshold / risk regime / 권장 포트폴리오 비중을 시각화하는 함수
    """

    required_cols = [
        date_col, var_col, exposure_col, reduce_signal_col,
        moderate_threshold_col, high_threshold_col, "risk_regime"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' 컬럼이 없습니다.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, var_col]).sort_values(date_col).reset_index(drop=True)

    moderate_mask = data["risk_regime"] == "MODERATE"
    high_mask = data["risk_regime"] == "HIGH"
    signal_mask = data[reduce_signal_col] == True

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Forecast VaR 선
    ax1.plot(
        data[date_col],
        data[var_col],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="예측 VaR"
    )

    # threshold 선
    ax1.axhline(
        data[moderate_threshold_col].iloc[0],
        color="orange",
        linestyle=":",
        linewidth=2.0,
        label="중간 리스크 기준선"
    )
    ax1.axhline(
        data[high_threshold_col].iloc[0],
        color="red",
        linestyle="-",
        linewidth=2.0,
        label="고위험 기준선"
    )

    # 리스크 구간 음영
    ax1.fill_between(
        data[date_col],
        data[var_col].min(),
        data[var_col].max(),
        where=moderate_mask,
        color="#FFD966",
        alpha=0.10,
        label="중간 리스크 구간"
    )

    ax1.fill_between(
        data[date_col],
        data[var_col].min(),
        data[var_col].max(),
        where=high_mask,
        color="#F4A6A6",
        alpha=0.14,
        label="고위험 구간"
    )

    # 비중 축소 신호
    ax1.scatter(
        data.loc[signal_mask, date_col],
        data.loc[signal_mask, var_col],
        color="darkred",
        s=45,
        marker="o",
        label="비중 축소 신호",
        zorder=5
    )

    ax1.set_title("Forecast VaR 기반 포트폴리오 비중 축소 기준")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("VaR 수익률")
    ax1.grid(True, alpha=0.3)

    # 보조축: 권장 포트폴리오 비중
    ax2 = ax1.twinx()
    ax2.plot(
        data[date_col],
        data[exposure_col],
        color="green",
        linewidth=1.8,
        label="권장 포트폴리오 비중"
    )
    ax2.set_ylabel("권장 포트폴리오 비중")
    ax2.set_ylim(0.0, 1.05)

    # 범례 합치기
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    handles = h1 + h2
    labels = l1 + l2

    desired_order = [
        "예측 VaR",
        "중간 리스크 기준선",
        "고위험 기준선",
        "중간 리스크 구간",
        "고위험 구간",
        "비중 축소 신호",
        "권장 포트폴리오 비중",
    ]

    ordered_handles = []
    ordered_labels = []

    for name in desired_order:
        if name in labels:
            idx = labels.index(name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    ax1.legend(
        ordered_handles,
        ordered_labels,
        loc="lower left",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.95,
        ncol=1
    )

    plt.tight_layout()

    if save_path:
        folder = save_path.rsplit("/", 1)[0]
        import os
        os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_garch_volatility_regime(
    regime_df: pd.DataFrame,
    save_path: str = "results/figures/garch_volatility_regime.png",
):
    """
    GARCH 조건부 변동성과 volatility regime 시각화

    Parameters
    ----------
    regime_df : pd.DataFrame
        compute_garch_volatility_regime() 결과 데이터프레임
    save_path : str
        그림 저장 경로
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = regime_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # -----------------------------
    # 1) 포트폴리오 수익률 (보조 정보)
    # -----------------------------
    ax1.plot(
        df["Date"],
        df["portfolio"],
        color="gray",
        alpha=0.35,
        linewidth=1.0,
        label="포트폴리오 수익률"
    )
    ax1.set_ylabel("포트폴리오 수익률")
    ax1.set_xlabel("Date")

    # -----------------------------
    # 2) Regime 배경색 표시
    # -----------------------------
    regime_colors = {
        "low": "#d9f2d9",
        "moderate": "#fff2cc",
        "high": "#f4cccc",
    }

    for i in range(len(df) - 1):
        reg = df.loc[i, "regime"]
        ax1.axvspan(
            df.loc[i, "Date"],
            df.loc[i + 1, "Date"],
            color=regime_colors.get(reg, "#ffffff"),
            alpha=0.35
        )

    # -----------------------------
    # 3) GARCH 변동성 (우측 축)
    # -----------------------------
    ax2 = ax1.twinx()

    ax2.plot(
        df["Date"],
        df["garch_vol"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="GARCH 조건부 변동성"
    )

    # 분위수 기준선
    low_thr = df["low_thr"].iloc[0]
    high_thr = df["high_thr"].iloc[0]

    ax2.axhline(
        low_thr,
        color="orange",
        linestyle=":",
        linewidth=2,
        label="중간 레짐 기준선"
    )
    ax2.axhline(
        high_thr,
        color="red",
        linestyle="-",
        linewidth=2,
        label="고변동성 기준선"
    )

    ax2.set_ylabel("GARCH 조건부 변동성")

    # -----------------------------
    # 4) High regime 구간 마커 표시
    # -----------------------------
    high_mask = df["regime"] == "high"
    ax2.scatter(
        df.loc[high_mask, "Date"],
        df.loc[high_mask, "garch_vol"],
        s=28,
        color="darkred",
        label="High Volatility Regime",
        zorder=5
    )

    # -----------------------------
    # 5) 제목 / 범례
    # -----------------------------
    plt.title("GARCH Volatility Regime of Portfolio")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")