import os  # 폴더 생성 등
import numpy as np  # 분위수(quantile) 계산용
import pandas as pd  # 데이터 처리

from src.config import CONF_LEVELS  # 예: [0.95, 0.99]
from scipy.stats import chi2


def load_returns(path: str = "data/processed/log_returns.csv") -> pd.DataFrame:
    """로그 수익률 CSV를 불러오는 함수"""
    rets = pd.read_csv(path, index_col=0, parse_dates=True)  # 날짜 인덱스 처리
    rets = rets.sort_index()  # 날짜 정렬
    return rets


def make_equal_weights(columns) -> pd.Series:
    """동일가중치(1/n) 생성 - 컬럼명 기준으로 안전하게 매핑"""
    n = len(columns)  # 종목 수
    w = pd.Series(1 / n, index=columns)  # 각 종목에 1/n 비중
    return w


def portfolio_return(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """포트폴리오 수익률 계산 (라벨 정렬 기반 dot)"""

    # 1️⃣ ticker 일치 여부 확인
    missing = set(returns.columns) - set(weights.index)
    extra = set(weights.index) - set(returns.columns)

    if missing:
        raise ValueError(f"weights에 없는 컬럼이 있습니다: {sorted(missing)}")
    if extra:
        raise ValueError(f"returns에 없는 ticker가 weights에 있습니다: {sorted(extra)}")

    # 2️⃣ 가중치 합이 1인지 확인 (허용 오차 1e-6)
    if abs(weights.loc[returns.columns].sum() - 1.0) > 1e-6:
        raise ValueError("가중치 합이 1이 아닙니다. weights를 확인하세요.")

    # 3️⃣ 컬럼 순서에 맞게 weight 정렬
    aligned_weights = weights.loc[returns.columns]

    # 4️⃣ 포트폴리오 수익률 계산
    port = returns.dot(aligned_weights)
    port.name = "portfolio"

    return port

def historical_var(x: pd.Series, alpha: float) -> float:
    """
    Historical VaR 계산
    alpha=0.95 -> 좌측 5% 분위수(손실 구간)
    """
    q = np.quantile(x, 1 - alpha)
    return float(q)


def cvar(x: pd.Series, alpha: float) -> float:
    """
    CVaR(Expected Shortfall) 계산
    VaR보다 더 나쁜 손실 구간의 평균
    """
    var = historical_var(x, alpha)
    tail = x[x <= var]
    return float(tail.mean())


def summarize_var_cvar(port: pd.Series, alphas=None) -> pd.DataFrame:
    """VaR/CVaR 요약 테이블 생성"""
    if alphas is None:
        alphas = CONF_LEVELS

    rows = []
    for a in alphas:
        rows.append({
            "alpha": a,
            "VaR": historical_var(port, a),
            "CVaR": cvar(port, a),
        })
    return pd.DataFrame(rows)

def rolling_historical_var(port_returns, window=252, alpha=0.95):
    """
    이동 Historical VaR 계산
    window: 이동 구간 길이 (예: 252일)
    alpha: 신뢰수준 (예: 0.95)
    """
    q = 1 - alpha

    rolling_var = (
        port_returns
        .rolling(window)
        .quantile(q)
    )

    return rolling_var

def kupiec_pof_test(
    violations: np.ndarray,
    expected_rate: float,
):
    """
    (KR) Kupiec POF(Proportion of Failures) Test
    - 목적: VaR 위반 비율이 이론적 기대치(p)와 일치하는지 통계적으로 검정

    H0(귀무가설): 실제 위반확률 = 기대 위반확률 p
    H1(대립가설): 실제 위반확률 != p

    Parameters
    ----------
    violations : np.ndarray (0/1 or False/True)
        위반 여부 배열 (1이면 위반)
    expected_rate : float
        기대 위반확률 (VaR 95% -> 0.05, VaR 99% -> 0.01)

    Returns
    -------
    dict:
        n: 총 관측치
        x: 위반 횟수
        fail_rate: 실제 위반율
        LR_pof: Kupiec 통계량 (chi-square(1))
        p_value: p-value
    """
    v = np.asarray(violations).astype(int)
    n = v.size
    x = int(v.sum())
    p = float(expected_rate)

    # 실제 위반율
    fail_rate = x / n if n > 0 else np.nan

    # 예외 처리: x=0 또는 x=n인 경우 로그 계산 안정화
    # (작은 epsilon 추가)
    eps = 1e-12
    phat = np.clip(fail_rate, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    # Kupiec LR statistic
    # LR = -2 ln( ( (1-p)^(n-x) p^x ) / ( (1-phat)^(n-x) phat^x ) )
    logL0 = (n - x) * np.log(1 - p) + x * np.log(p)
    logL1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    LR_pof = -2 * (logL0 - logL1)

    # p-value (chi-square df=1)
    p_value = 1 - chi2.cdf(LR_pof, df=1)

    return {
        "n": n,
        "x": x,
        "fail_rate": fail_rate,
        "LR_pof": LR_pof,
        "p_value": p_value,
    }


def run_var_backtest_kupiec(df, level: int):
    """
    (KR) 특정 신뢰수준(level)에 대해
    - 위반율 계산
    - Kupiec POF 테스트 수행
    """
    vio_col = f"Violation_{level}"
    if vio_col not in df.columns:
        raise ValueError(f"필요 컬럼 없음: {vio_col}")

    expected_rate = 0.05 if level == 95 else 0.01
    result = kupiec_pof_test(df[vio_col].values, expected_rate)

    # (KR) 결과 요약 문장 (Velog/README에 바로 쓰기 좋게)
    verdict = "PASS" if result["p_value"] > 0.05 else "FAIL"

    summary = {
        "level": level,
        "expected_rate": expected_rate,
        "n": result["n"],
        "violations": result["x"],
        "violation_rate": result["fail_rate"],
        "LR_pof": result["LR_pof"],
        "p_value": result["p_value"],
        "verdict": verdict,
    }
    return summary

if __name__ == "__main__":
    # 결과 저장 폴더 생성
    os.makedirs("results/tables", exist_ok=True)

    # 1) 로그 수익률 로드
    rets = load_returns()

    # 2) 동일가중치 생성
    w = make_equal_weights(rets.columns)

    # 3) 포트폴리오 수익률 계산
    port = portfolio_return(rets, w)

    # 3-1) 분포 진단: 왜도/첨도 계산 (분포의 비대칭성과 꼬리 두께 확인)
    skew = float(port.skew())        # 왜도(스큐니스): 0이면 대칭, 음수면 좌측 꼬리(손실) 쪽이 더 김
    kurt = float(port.kurt())        # 첨도(excess kurtosis): 0이면 정규분포와 동일, 양수면 fat-tail

    print(f"Skewness(왜도): {skew:.4f}")
    print(f"Kurtosis(첨도, excess): {kurt:.4f}")

    # 3-2) 분포 진단 결과 저장 (리포트/README용)
    dist_stats = pd.DataFrame([{
        "skewness": skew,
        "kurtosis_excess": kurt
    }])
    dist_stats.to_csv("results/tables/dist_stats.csv", index=False)
    print("Saved: results/tables/dist_stats.csv")

    # 4) VaR/CVaR 요약
    summary = summarize_var_cvar(port)

    # 4-1) Stress Testing (시나리오 기반 손실 점검)
    # - 확률을 추정하는 것이 아니라, "만약 이런 충격이 오면?"을 계산
    stress_scenarios = {
        "Mild Shock (-5%)": -0.05,
        "Severe Shock (-10%)": -0.10,
        "Crisis Shock (-20%)": -0.20
    }

    portfolio_value = 10_000_000  # 예시: 1천만 원 (원하는 값으로 변경)

    stress_rows = []
    for name, shock in stress_scenarios.items():
        loss_amount = portfolio_value * abs(shock)  # 손실 금액 계산

        stress_rows.append({
            "scenario": name,
            "shock_return": shock,
            "loss_krw": loss_amount
        })

    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv("results/tables/stress_test.csv", index=False)
    print("Saved: results/tables/stress_test.csv")
    print(stress_df)

    # 5) 저장
    out_path = "results/tables/var_summary.csv"
    summary.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(summary)

    # 6) 이동 VaR 계산
    # Load log return data
    df = pd.read_csv("data/processed/log_returns.csv")

    returns = df.drop(columns=["Date"], errors="ignore").dropna().astype(float)

    # equal-weight portfolio
    port = returns.mean(axis=1)
    
    rolling_var_95 = rolling_historical_var(port, window=252, alpha=0.95)
    rolling_var_99 = rolling_historical_var(port, window=252, alpha=0.99)

    rolling_df = pd.DataFrame({
        "Date": df["Date"],
        "Rolling_VaR_95": rolling_var_95,
        "Rolling_VaR_99": rolling_var_99,
    })

    rolling_df = rolling_df.dropna().reset_index(drop=True)

    rolling_df.to_csv("results/tables/rolling_var.csv", index=False)
    print("Saved: results/tables/rolling_var.csv")