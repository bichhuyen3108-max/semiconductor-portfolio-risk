import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    port = returns.mean(axis=1)

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

        var = np.quantile(port, q)
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

    # 동일가중 포트폴리오 수익률
    port = returns.mean(axis=1)

    # 포트폴리오 수익률 DF
    port_df = pd.DataFrame({
        "Date": log_df.loc[returns.index, "Date"].values,
        "Portfolio_Return": port.values
    })

    # 2) Rolling VaR 로드
    rolling_df = pd.read_csv(rolling_var_path)
    if "Date" not in rolling_df.columns:
        raise ValueError("rolling_var.csv 파일에 'Date' 컬럼이 없습니다.")

    rolling_df["Date"] = pd.to_datetime(rolling_df["Date"])

    # 3) Date 기준으로 결합 (inner join: 공통 날짜만)
    df = pd.merge(rolling_df, port_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    # 필수 컬럼 체크
    for col in ["Rolling_VaR_95", "Rolling_VaR_99"]:
        if col not in df.columns:
            raise ValueError(f"rolling_var.csv 파일에 '{col}' 컬럼이 없습니다.")

    return df


def add_violations(df: pd.DataFrame) -> pd.DataFrame:
    """
    (KR) 위반(Violation) 생성: 실제 수익률 < VaR 인 경우 True
    """
    out = df.copy()
    out["Violation_95"] = out["Portfolio_Return"] < out["Rolling_VaR_95"]
    out["Violation_99"] = out["Portfolio_Return"] < out["Rolling_VaR_99"]
    return out


def plot_return_vs_rolling_var(
    df: pd.DataFrame,
    level: int = 95,
    save_path: str | None = "results/figures/rolling_var_violation_95.png",
    show: bool = True,
):
    """
    (KR) 포트폴리오 수익률 시계열 + 이동 VaR 오버레이 + 위반(빨간 점) 표시
    (VN) Vẽ time series return + đường rolling VaR + đánh dấu violation (chấm đỏ)

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
    plt.plot(df["Date"], df["Portfolio_Return"], label="Portfolio Return", alpha=0.6)

    # Rolling VaR
    plt.plot(df["Date"], df[var_col], label=f"Rolling VaR {level}%", linestyle="--")

    # 위반 포인트 강조
    vio_points = df[df[vio_col]]
    plt.scatter(
        vio_points["Date"],
        vio_points["Portfolio_Return"],
        label=f"Violation ({level}%)",
        s=18
    )

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


def run_rolling_var_plots():
    """
    (KR) 전체 실행: 데이터 로드 -> 위반 생성 -> 95%, 99% 그래프 저장/표시
    """
    df = load_portfolio_and_rolling_var()
    df = add_violations(df)

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