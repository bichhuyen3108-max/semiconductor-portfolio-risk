import os
import numpy as np
import pandas as pd

from .config import CONF_LEVELS
from .portfolio import make_equal_weights, portfolio_return
from scipy.stats import chi2, t, norm
from arch import arch_model


def load_returns(path: str = "data/processed/log_returns.csv") -> pd.DataFrame:
    """로그 수익률 CSV를 불러오는 함수"""
    rets = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    rets = rets.apply(pd.to_numeric, errors="coerce")
    rets = rets.dropna(thresh=int(len(rets.columns) * 0.6))  
    return rets


def historical_var(x: pd.Series, alpha: float) -> float:
    """Historical VaR: alpha=0.95 -> 5% left tail quantile"""
    return float(np.quantile(x, 1 - alpha))


def cvar(x: pd.Series, alpha: float) -> float:
    """CVaR (Expected Shortfall): mean of returns <= VaR"""
    var = historical_var(x, alpha)
    tail = x[x <= var]
    return float(tail.mean())


def summarize_var_cvar(port: pd.Series, alphas=None) -> pd.DataFrame:
    """VaR/CVaR 요약 테이블 생성"""
    if alphas is None:
        alphas = CONF_LEVELS

    rows = []
    for a in alphas:
        rows.append({"alpha": a, "VaR": historical_var(port, a), "CVaR": cvar(port, a)})
    return pd.DataFrame(rows)


def rolling_historical_var(port_returns: pd.Series, window=252, alpha=0.95) -> pd.Series:
    """이동 Historical VaR"""
    return port_returns.rolling(window).quantile(1 - alpha)

def rolling_student_t_var(port_returns: pd.Series, window=252, alpha=0.95) -> pd.Series:
    """
    Rolling Student-t VaR (t분포 VaR)
    - Fit Student-t on rolling window, take left tail quantile (1-alpha)
    - Convention: VaR is negative threshold, violation if return < VaR
    """
    tail_p = 1 - alpha
    out = np.full(len(port_returns), np.nan)

    for i in range(window, len(port_returns)):
        w = port_returns.iloc[i-window:i].dropna().values
        if len(w) < window * 0.9:
            continue

        # fit: df, loc, scale
        df_hat, loc_hat, scale_hat = t.fit(w)
        out[i] = t.ppf(tail_p, df_hat, loc=loc_hat, scale=scale_hat)

    return pd.Series(out, index=port_returns.index)


def rolling_garch_var(
    port_returns: pd.Series,
    window: int = 252,
    alpha: float = 0.95,
    dist: str = "normal",
    refit_every: int = 20
) -> pd.Series:
    """
    Rolling GARCH(1,1) 기반 VaR 계산 함수

    설명:
    - 과거 window 기간 데이터를 이용하여 GARCH 모델을 추정
    - 1-step ahead 변동성 예측
    - VaR = μ + σ * q (좌측 tail quantile)

    안정성과 계산 효율성을 위해
    refit_every 거래일마다 GARCH 모델을 재추정하고
    그 사이 구간은 최근 추정된 모델을 사용하여 예측

    dist:
    - "normal" : 정규분포 오차
    - "t" : Student-t 분포 오차
    """

    # VaR 좌측 tail 확률
    tail_p = 1 - alpha

    # 결과 저장 배열
    out = np.full(len(port_returns), np.nan)

    # arch 패키지는 % 단위 데이터에서 더 안정적으로 추정됨
    r_pct = port_returns * 100

    # normal 분포의 quantile 미리 계산
    if dist == "normal":
        q = float(norm.ppf(tail_p))
    else:
        q = None

    # 이전에 추정된 GARCH 모델 저장
    last_res = None

    for i in range(window, len(r_pct)):

        # 일정 기간마다 모델 재추정
        need_refit = (
            last_res is None
            or (refit_every is not None and (i - window) % refit_every == 0)
        )

        if need_refit:

            # rolling window 데이터
            w = r_pct.iloc[i - window:i].dropna()

            # 데이터가 충분하지 않으면 skip
            if len(w) < window * 0.9:
                continue

            try:
                # GARCH(1,1) 모델 추정
                am = arch_model(
                    w,
                    vol="Garch",
                    p=1,
                    q=1,
                    mean="Constant",
                    dist=dist
                )

                last_res = am.fit(disp="off")

            except Exception:
                # 수렴 실패 시 해당 시점 skip
                last_res = None
                continue

        try:
            # 1-step ahead 변동성 예측
            f = last_res.forecast(horizon=1, reindex=False)

            sigma2 = float(f.variance.values[-1, 0])

            # 비정상 값 제거
            if not np.isfinite(sigma2) or sigma2 <= 0:
                continue

            sigma = float(np.sqrt(sigma2))

            # 평균 수익률
            mu = float(last_res.params.get("mu", 0.0))

            # Student-t 분포일 경우 자유도 사용
            if dist == "t":
                nu = float(last_res.params.get("nu"))
                q = float(t.ppf(tail_p, df=nu))

            # VaR 계산
            var_pct = mu + sigma * q

            # 다시 decimal return으로 변환
            out[i] = var_pct / 100.0

        except Exception:
            continue

    return pd.Series(out, index=port_returns.index)


def kupiec_pof_test(violations: np.ndarray, expected_rate: float):
    """Kupiec POF Test"""
    v = np.asarray(violations).astype(int)
    n = v.size
    x = int(v.sum())
    p = float(expected_rate)

    fail_rate = x / n if n > 0 else np.nan

    eps = 1e-12
    phat = np.clip(fail_rate, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    logL0 = (n - x) * np.log(1 - p) + x * np.log(p)
    logL1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    LR_pof = -2 * (logL0 - logL1)

    p_value = 1 - chi2.cdf(LR_pof, df=1)

    return {"n": n, "x": x, "fail_rate": fail_rate, "LR_pof": LR_pof, "p_value": p_value}


def build_rolling_var_and_violations(
    port: pd.Series,
    window: int = 252,
    alphas=(0.95, 0.99),
) -> pd.DataFrame:
    """
    rolling VaR + violation columns 생성
    Violation: 실제 수익률 < VaR (더 나쁨) 이면 True
    """
    out = pd.DataFrame({"Date": port.index, "portfolio": port.values})

    for a in alphas:
        level = int(a * 100)
        var_col = f"Rolling_VaR_{level}"
        vio_col = f"Violation_{level}"

        out[var_col] = rolling_historical_var(port, window=window, alpha=a).values
        out[vio_col] = out["portfolio"] < out[var_col]

    out = out.dropna().reset_index(drop=True)
    return out

def build_rolling_var_models_and_violations(
    port: pd.Series,
    window: int = 252,
    alphas=(0.95, 0.99),
    garch_dist: str = "normal"
) -> pd.DataFrame:
    """
    Student-t VaR + GARCH VaR rolling + violation columns
    Output: rolling_var_models.csv
    """
    out = pd.DataFrame({"Date": port.index, "portfolio": port.values})

    for a in alphas:
        level = int(a * 100)

        # 1) Student-t
        tvar_col = f"Rolling_tVaR_{level}"
        tvio_col = f"Violation_tVaR_{level}"
        out[tvar_col] = rolling_student_t_var(port, window=window, alpha=a).values
        out[tvio_col] = out["portfolio"] < out[tvar_col]

        # 2) GARCH
        gvar_col = f"Rolling_GARCHVaR_{level}"
        gvio_col = f"Violation_GARCHVaR_{level}"
        out[gvar_col] = rolling_garch_var(port, window=window, alpha=a, dist=garch_dist, refit_every=5).values
        out[gvio_col] = out["portfolio"] < out[gvar_col]

    needed_cols = ["portfolio"]
    for a in alphas:
        level = int(a * 100)
        needed_cols += [f"Rolling_tVaR_{level}", f"Rolling_GARCHVaR_{level}"]

    out = out.dropna(subset=needed_cols).reset_index(drop=True)
    return out
  


def run_var_backtest_kupiec(df: pd.DataFrame, level: int):
    """특정 신뢰수준(level) Kupiec POF 테스트"""
    vio_col = f"Violation_{level}"
    if vio_col not in df.columns:
        raise ValueError(f"필요 컬럼 없음: {vio_col}")

    expected_rate = 0.05 if level == 95 else 0.01
    result = kupiec_pof_test(df[vio_col].values, expected_rate)

    verdict = "PASS" if result["p_value"] > 0.05 else "FAIL"

    return {
        "level": level,
        "expected_rate": expected_rate,
        "n": result["n"],
        "violations": result["x"],
        "violation_rate": result["fail_rate"],
        "LR_pof": result["LR_pof"],
        "p_value": result["p_value"],
        "verdict": verdict,
    }

def run_var_backtest_kupiec_by_col(df: pd.DataFrame, vio_col: str, level: int):
    """Kupiec POF test for any violation column"""
    if vio_col not in df.columns:
        raise ValueError(f"필요 컬럼 없음: {vio_col}")

    expected_rate = 0.05 if level == 95 else 0.01
    result = kupiec_pof_test(df[vio_col].values, expected_rate)
    verdict = "PASS" if result["p_value"] > 0.05 else "FAIL"

    return {
        "model": vio_col.replace("Violation_", "").replace(f"_{level}", ""),
        "level": level,
        "expected_rate": expected_rate,
        "n": result["n"],
        "violations": result["x"],
        "violation_rate": result["fail_rate"],
        "LR_pof": result["LR_pof"],
        "p_value": result["p_value"],
        "verdict": verdict,
    }


def estimate_risk_thresholds_from_forecast_var(
    merged_df: pd.DataFrame,
    var_col: str = "VaR_return_h",
    violation_col: str = "violation",
) -> dict:
    """
    Forecast VaR 분포와 violation 데이터를 기반으로
    리스크 threshold를 추정하는 함수

    기준
    ----------
    - moderate_threshold: Forecast VaR 분포의 80% 분위수
    - high_threshold: violation이 발생한 구간의 Forecast VaR 중앙값(median)

    해석
    ----------
    Forecast VaR는 음수이므로,
    값이 더 작을수록(더 음수일수록) 리스크가 크다고 해석한다.
    """

    if var_col not in merged_df.columns:
        raise ValueError(f"'{var_col}' 컬럼이 없습니다.")
    if violation_col not in merged_df.columns:
        raise ValueError(f"'{violation_col}' 컬럼이 없습니다.")

    df = merged_df.copy()
    df[var_col] = pd.to_numeric(df[var_col], errors="coerce")
    df = df.dropna(subset=[var_col])

    # 전체 Forecast VaR 분포 기준 중간 리스크 임계값
    moderate_threshold = df[var_col].quantile(0.20)
    # 음수 값에서 20% 분위수는 더 '낮은 값' = 더 큰 리스크 구간

    # violation 발생 구간의 VaR 분포를 따로 확인
    vio_df = df[df[violation_col] == True].copy()

    if len(vio_df) == 0:
        # violation이 전혀 없다면 보수적으로 전체 분포의 10% 분위수 사용
        high_threshold = df[var_col].quantile(0.10)
    else:
        # violation이 발생한 구간에서의 VaR 수준을 high risk 기준으로 사용
        high_threshold = vio_df[var_col].median()

    # VaR는 음수이므로, 더 작은 값(더 음수)이 더 높은 리스크를 의미함
    # 따라서 high_threshold는 반드시 moderate_threshold보다 작아야 함
    if high_threshold >= moderate_threshold:
        moderate_threshold, high_threshold = high_threshold, moderate_threshold
        
    return {
        "moderate_threshold": float(moderate_threshold),
        "high_threshold": float(high_threshold),
        "n_total": int(len(df)),
        "n_violation": int(df[violation_col].sum()),
        "violation_rate": float(df[violation_col].mean()),
    }


def add_risk_regime_from_forecast_var(
    merged_df: pd.DataFrame,
    var_col: str = "VaR_return_h",
    threshold_info: dict = None,
    base_exposure: float = 1.0,
    moderate_exposure: float = 0.8,
    high_exposure: float = 0.6,
) -> pd.DataFrame:
    """
    Forecast VaR threshold를 기반으로
    LOW / MODERATE / HIGH 리스크 레짐과 권장 포트폴리오 비중을 생성하는 함수
    """

    if var_col not in merged_df.columns:
        raise ValueError(f"'{var_col}' 컬럼이 없습니다.")
    if threshold_info is None:
        raise ValueError("threshold_info가 필요합니다.")

    moderate_threshold = threshold_info["moderate_threshold"]
    high_threshold = threshold_info["high_threshold"]

    df = merged_df.copy()
    df[var_col] = pd.to_numeric(df[var_col], errors="coerce")
    df = df.dropna(subset=[var_col]).reset_index(drop=True)

    # 기본값 설정
    df["risk_regime"] = "LOW"
    df["suggested_exposure"] = base_exposure

    # Forecast VaR가 threshold보다 더 작아질수록 리스크가 높다고 판단
    moderate_mask = (df[var_col] <= moderate_threshold) & (df[var_col] > high_threshold)
    high_mask = df[var_col] <= high_threshold

    df.loc[moderate_mask, "risk_regime"] = "MODERATE"
    df.loc[moderate_mask, "suggested_exposure"] = moderate_exposure

    df.loc[high_mask, "risk_regime"] = "HIGH"
    df.loc[high_mask, "suggested_exposure"] = high_exposure

    # 전일 대비 권장 비중이 감소한 시점을 비중 축소 신호로 정의
    df["reduce_signal"] = df["suggested_exposure"].diff().fillna(0) < 0

    # threshold 값 저장
    df["moderate_threshold"] = moderate_threshold
    df["high_threshold"] = high_threshold

    return df


def save_risk_threshold_summary(
    threshold_info: dict,
    save_path: str = "results/tables/risk_threshold_summary.csv"
) -> pd.DataFrame:
    """
    추정된 risk threshold 정보를 csv로 저장하는 함수
    """
    out = pd.DataFrame([threshold_info])
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {save_path}")
    return out


if __name__ == "__main__":

    print("RISK_METRICS PATH:", __file__)
    os.makedirs("results/tables", exist_ok=True)

    # 1) returns 로드
    rets = load_returns()

    # 2) portfolio return (portfolio.py 사용)
    w = make_equal_weights(rets.columns)
    port = portfolio_return(rets, w)

    # 3) 분포 진단 (skew/kurt)
    dist_stats = pd.DataFrame([{
        "skewness": float(port.skew()),
        "kurtosis_excess": float(port.kurt()),
    }])
    dist_stats.to_csv("results/tables/dist_stats.csv", index=False)

    # 4) VaR/CVaR summary
    summary = summarize_var_cvar(port)
    summary.to_csv("results/tables/var_summary.csv", index=False)

    # 5) Stress test
    stress_scenarios = {
        "Mild Shock (-5%)": -0.05,
        "Severe Shock (-10%)": -0.10,
        "Crisis Shock (-20%)": -0.20,
    }
    portfolio_value_krw = 10_000_000

    stress_df = pd.DataFrame([{
        "scenario": name,
        "shock_return": shock,
        "loss_krw": portfolio_value_krw * abs(shock),
    } for name, shock in stress_scenarios.items()])
    stress_df.to_csv("results/tables/stress_test.csv", index=False)

    # 6-1) Rolling VaR + Violations + Kupiec
    rolling_df = build_rolling_var_and_violations(port, window=252, alphas=(0.95, 0.99))
    rolling_df.to_csv("results/tables/rolling_var.csv", index=False)

    kupiec_95 = run_var_backtest_kupiec(rolling_df, 95)
    kupiec_99 = run_var_backtest_kupiec(rolling_df, 99)
    kupiec_df = pd.DataFrame([kupiec_95, kupiec_99])
    kupiec_df.to_csv("results/tables/kupiec_test.csv", index=False)

    print("Saved: results/tables/dist_stats.csv")
    print("Saved: results/tables/var_summary.csv")
    print("Saved: results/tables/stress_test.csv")
    print("Saved: results/tables/rolling_var.csv")
    print("Saved: results/tables/kupiec_test.csv")
    print(kupiec_df)


    # 6-2) Rolling VaR Models (Student-t, GARCH) + Violations

    rolling_models_df = build_rolling_var_models_and_violations(
        port, window=252, alphas=(0.95, 0.99), garch_dist="normal"
    )
    rolling_models_df.to_csv("results/tables/rolling_var_models.csv", index=False)

    # 6-3) Kupiec backtest for models
    rows = []
    for level in [95, 99]:
        rows.append(run_var_backtest_kupiec_by_col(rolling_models_df, f"Violation_tVaR_{level}", level))
        rows.append(run_var_backtest_kupiec_by_col(rolling_models_df, f"Violation_GARCHVaR_{level}", level))

    kupiec_models_df = pd.DataFrame(rows)
    kupiec_models_df.to_csv("results/tables/kupiec_models.csv", index=False)

    print("Saved: results/tables/rolling_var_models.csv")
    print("Saved: results/tables/kupiec_models.csv")
    print(kupiec_models_df)


