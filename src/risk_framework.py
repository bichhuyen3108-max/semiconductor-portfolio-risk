import os
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

from .risk_metrics import (
    estimate_risk_thresholds_from_forecast_var,
    add_risk_regime_from_forecast_var,
    save_risk_threshold_summary,
)
from .visualization import plot_forecast_var_risk_threshold_signal

from arch import arch_model
from src.portfolio import make_equal_weights, portfolio_return

# =========================================================
# 1) Stress Scenarios
# =========================================================
def run_stress_scenarios(
    df: pd.DataFrame,
    portfolio_value: float = 10_000_000,
    ret_col: str = "portfolio",
    scenarios: Optional[Dict[str, float]] = None,
    include_historical_worst: bool = True,
) -> pd.DataFrame:
    """
    Stress scenario 기반 손익(PnL) 및 손실 규모(Loss) 추정 (KRW).
    - PnL_KRW: 음수=손실, 양수=이익
    - Loss_KRW: 손실 규모(항상 양수)
    """
    if ret_col not in df.columns:
        raise ValueError(f"df에 '{ret_col}' 컬럼이 없습니다.")

    base = {
        "Moderate Shock (-5%)": -0.05,
        "Severe Shock (-10%)": -0.10,
    }
    if scenarios:
        base.update(scenarios)

    if include_historical_worst:
        worst = float(pd.to_numeric(df[ret_col], errors="coerce").min())
        base["Historical Worst (1D Return)"] = worst

    rows = []
    for name, shock in base.items():
        pnl = float(portfolio_value) * shock
        rows.append({
            "Scenario": name,
            "Shock_Return": shock,
            "PnL_KRW": pnl,                 # 음수=손실
            "Loss_KRW": abs(pnl),           # 손실 규모(양수)
        })

    return pd.DataFrame(rows).sort_values("Shock_Return").reset_index(drop=True)


# =========================================================
# 2) VaR Forecast (single point + series)
# =========================================================
def forecast_var(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    lookback_days: int = 30,
    horizon_days: int = 5,
    portfolio_value: float = 10_000_000,
    ret_col: str = "portfolio",
    assume_mean_zero: bool = True,
) -> Dict[str, Any]:
    """
    최근 lookback_days 변동성 기반 Parametric Normal VaR (horizon_days).
    """
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level은 0과 1 사이여야 합니다. 예: 0.95")
    if lookback_days < 5:
        raise ValueError("lookback_days는 최소 5 이상 권장입니다.")
    if horizon_days < 1:
        raise ValueError("horizon_days는 1 이상이어야 합니다.")
    if ret_col not in df.columns:
        raise ValueError(f"df에 '{ret_col}' 컬럼이 없습니다.")

    s = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if len(s) < lookback_days:
        raise ValueError(f"데이터가 부족합니다. 최소 {lookback_days}개 이상 필요합니다.")

    window = s.iloc[-lookback_days:]

    sigma_1d = float(window.std(ddof=1))
    sigma_h = sigma_1d * float(np.sqrt(horizon_days))

    if assume_mean_zero:
        mu_h = 0.0
    else:
        mu_1d = float(window.mean())
        mu_h = mu_1d * float(horizon_days)

    left_tail_prob = 1.0 - float(confidence_level)
    q_left = float(norm.ppf(left_tail_prob))  # 음수

    var_return = mu_h + q_left * sigma_h
    var_krw = float(portfolio_value) * var_return

    return {
        "confidence_level": float(confidence_level),
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "assume_mean_zero": bool(assume_mean_zero),
        "sigma_1d": sigma_1d,
        "sigma_h": sigma_h,
        "left_tail_prob": left_tail_prob,
        "q_left": q_left,
        "var_return": var_return,
        "var_krw": var_krw,
    }


def build_var_forecast_series(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    lookback_days: int = 30,
    horizon_days: int = 5,
    portfolio_value: float = 10_000_000,
    date_col: str = "Date",
    ret_col: str = "portfolio",
    assume_mean_zero: bool = True,
) -> pd.DataFrame:
    """
    날짜별 VaR 예측 시계열 생성.
    반환 컬럼: Date, sigma_1d, sigma_h, q_left, VaR_return_h, VaR_krw_h
    """
    if date_col not in df.columns:
        raise ValueError(f"df에 '{date_col}' 컬럼이 없습니다.")
    if ret_col not in df.columns:
        raise ValueError(f"df에 '{ret_col}' 컬럼이 없습니다.")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level은 0과 1 사이여야 합니다.")
    if lookback_days < 5:
        raise ValueError("lookback_days는 최소 5 이상 권장입니다.")
    if horizon_days < 1:
        raise ValueError("horizon_days는 1 이상이어야 합니다.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")
    out = out.dropna(subset=[date_col, ret_col]).sort_values(date_col).reset_index(drop=True)

    out["sigma_1d"] = out[ret_col].rolling(lookback_days).std(ddof=1)
    out["sigma_h"] = out["sigma_1d"] * np.sqrt(horizon_days)

    if assume_mean_zero:
        out["mu_h"] = 0.0
    else:
        out["mu_1d"] = out[ret_col].rolling(lookback_days).mean()
        out["mu_h"] = out["mu_1d"] * horizon_days

    q_left = float(norm.ppf(1.0 - confidence_level))  # 음수
    out["q_left"] = q_left

    out["VaR_return_h"] = out["mu_h"] + out["q_left"] * out["sigma_h"]
    out["VaR_krw_h"] = float(portfolio_value) * out["VaR_return_h"]

    cols = [date_col, "sigma_1d", "sigma_h", "q_left", "VaR_return_h", "VaR_krw_h"]
    return out[cols].dropna().reset_index(drop=True)


# =========================================================
# 3) Stop-loss Decision (used in runner)
# =========================================================
def stop_loss_decision_dynamic(
    forecast_result: Dict[str, Any],
    portfolio_value: float = 10_000_000,
    risk_budget_pct: float = 0.08,
    moderate_ratio: float = 0.6,
) -> Dict[str, Any]:
    """
    Risk budget 기반 의사결정.
    - loss_krw = abs(VaR_krw)
    - budget_krw = portfolio_value * risk_budget_pct
    """
    var_krw = float(forecast_result["var_krw"])  # 보통 음수
    loss_krw = abs(var_krw)

    budget_krw = float(portfolio_value) * float(risk_budget_pct)
    moderate_krw = budget_krw * float(moderate_ratio)

    if loss_krw >= budget_krw:
        level = "HIGH"
        action = "노출 축소(예: 20%) 및 신규 매수 보류"
    elif loss_krw >= moderate_krw:
        level = "MODERATE"
        action = "변동성 모니터링 강화 및 부분 축소 검토"
    else:
        level = "LOW"
        action = "포지션 유지(정상 범위)"

    return {
        "rule_type": "RISK_BUDGET",
        "Risk_Level": level,
        "Action": action,
        "risk_budget_pct": float(risk_budget_pct),
        "risk_budget_krw": budget_krw,
        "estimated_var_loss_krw": loss_krw,
    }


# =========================================================
# 4) Realized horizon return + plots
# =========================================================
def add_realized_horizon_return(
    df: pd.DataFrame,
    horizon_days: int = 5,
    date_col: str = "Date",
    ret_col: str = "portfolio",
) -> pd.DataFrame:
    """
    (t+1 ~ t+horizon_days) 구간의 누적 로그수익률
    (log return은 합으로 누적 가능)

    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")
    out = out.dropna(subset=[date_col, ret_col]).sort_values(date_col).reset_index(drop=True)

    future_sum = None
    for k in range(1, horizon_days + 1):
        term = out[ret_col].shift(-k)
        future_sum = term if future_sum is None else (future_sum + term)

    out["realized_return_h"] = future_sum
    return out[[date_col, "realized_return_h"]].dropna().reset_index(drop=True)


def _safe_savefig(save_path: Optional[str]) -> None:
    if not save_path:
        return
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved figure: {save_path}")


def plot_var_with_regime_shading(
    var_series: pd.DataFrame,
    risk_budget_pct: float = 0.08,
    moderate_ratio: float = 0.6,
    date_col: str = "Date",
    var_col: str = "VaR_return_h",
    save_path: Optional[str] = "results/figures/post4_var_regime_shading.png",
    show: bool = True,
) -> None:
    """
    Forecast VaR + Risk Budget Line + 위험 구간(LOW/MODERATE/HIGH) 음영 처리
    """
    if date_col not in var_series.columns or var_col not in var_series.columns:
        raise ValueError("var_series에 필요한 컬럼이 없습니다.")

    
    dfp = var_series.sort_values(date_col).reset_index(drop=True)
    dfp[date_col] = pd.to_datetime(dfp[date_col], errors="coerce")
    dfp = dfp.dropna(subset=[date_col, var_col])

    budget = -float(risk_budget_pct)
    moderate = budget * float(moderate_ratio)  # 예: -0.08 * 0.6 = -0.048

    cond_high = dfp[var_col] <= budget
    cond_mod = (dfp[var_col] > budget) & (dfp[var_col] <= moderate)
    cond_low = dfp[var_col] > moderate

    plt.figure(figsize=(14, 6))
    plt.plot(dfp[date_col], dfp[var_col], label="Forecast VaR (horizon)", linestyle="--")
    plt.axhline(budget, label=f"Risk Budget Line ({-budget:.0%} loss)", linewidth=2)

    ax = plt.gca()
    ax.fill_between(dfp[date_col], dfp[var_col], budget, where=cond_low, alpha=0.08, label="LOW")
    ax.fill_between(dfp[date_col], dfp[var_col], budget, where=cond_mod, alpha=0.16, label="MODERATE")
    ax.fill_between(dfp[date_col], dfp[var_col], budget, where=cond_high, alpha=0.28, label="HIGH")

    plt.title("Forecast VaR with Risk Regime Shading (LOW / MODERATE / HIGH)")
    plt.xlabel("Date")
    plt.ylabel("VaR Return")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    _safe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_forecast_var_vs_realized_return(
    var_series: pd.DataFrame,
    realized_df: pd.DataFrame,
    date_col: str = "Date",
    var_col: str = "VaR_return_h",
    realized_col: str = "realized_return_h",
    save_path: Optional[str] = "results/figures/post4_forecast_vs_realized.png",
    show: bool = True,
) -> pd.DataFrame:
    """
    Forecast VaR(horizon) vs Realized horizon return 비교 + violation 표시
    violation: realized_return_h < VaR_return_h
    """
    if date_col not in var_series.columns or var_col not in var_series.columns:
        raise ValueError("var_series에 필요한 컬럼이 없습니다.")
    if date_col not in realized_df.columns or realized_col not in realized_df.columns:
        raise ValueError("realized_df에 필요한 컬럼이 없습니다.")
    
    # Ensure datetime BEFORE merge
    var_series = var_series.copy()
    realized_df = realized_df.copy()

    var_series[date_col] = pd.to_datetime(var_series[date_col], errors="coerce")
    realized_df[date_col] = pd.to_datetime(realized_df[date_col], errors="coerce")

    var_series = var_series.dropna(subset=[date_col])
    realized_df = realized_df.dropna(subset=[date_col])

    dfm = pd.merge(
        var_series[[date_col, var_col]],
        realized_df[[date_col, realized_col]],
        on=date_col,
        how="inner",
    ).sort_values(date_col).reset_index(drop=True)

    dfm[date_col] = pd.to_datetime(dfm[date_col], errors="coerce")
    dfm = dfm.dropna(subset=[date_col, var_col, realized_col])

    dfm["violation"] = dfm[realized_col] < dfm[var_col]
    vio_rate = float(dfm["violation"].mean())
    print(f"[INFO] Horizon violation rate = {vio_rate:.4f} ({int(dfm['violation'].sum())}/{len(dfm)})")


    plt.figure(figsize=(14, 6))
    plt.plot(dfm[date_col], dfm[var_col], label="Forecast VaR (horizon)", linestyle="--")
    plt.plot(dfm[date_col], dfm[realized_col], label="Realized Return (horizon)")

    vio = dfm[dfm["violation"]]
    plt.scatter(vio[date_col], vio[realized_col], label="Violation", s=18)

    plt.title("Forecast VaR vs Realized Horizon Return (Violation Overlay)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    _safe_savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

    return dfm


# =========================================================
# 5) Risk Threshold from Forecast VaR + Violation
# =========================================================
def run_post4_risk_threshold_analysis(
    merged_df: pd.DataFrame,
    save_tables_dir: str = "results/tables",
    save_figures_dir: str = "results/figures",
    show_plot: bool = True,
):
    """
    Forecast VaR와 violation 분석 결과를 기반으로
    리스크 threshold를 추정하고,
    포트폴리오 비중 축소 신호를 생성 및 시각화하는 함수
    """

    os.makedirs(save_tables_dir, exist_ok=True)
    os.makedirs(save_figures_dir, exist_ok=True)

    # 1. Forecast VaR 분포와 violation 정보를 이용하여 threshold 추정
    threshold_info = estimate_risk_thresholds_from_forecast_var(
        merged_df,
        var_col="VaR_return_h",
        violation_col="violation",
    )

    # 2. threshold 요약 저장
    save_risk_threshold_summary(
        threshold_info,
        save_path=os.path.join(save_tables_dir, "post4_risk_threshold_summary.csv")
    )

    # 3. threshold를 기반으로 리스크 레짐 및 권장 포트폴리오 비중 생성
    signal_df = add_risk_regime_from_forecast_var(
        merged_df,
        var_col="VaR_return_h",
        threshold_info=threshold_info,
        base_exposure=1.0,
        moderate_exposure=0.8,
        high_exposure=0.6,
    )

    # 4. 결과 테이블 저장
    signal_path = os.path.join(save_tables_dir, "post4_risk_threshold_signal.csv")
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {signal_path}")

    # 5. 시각화 저장
    plot_forecast_var_risk_threshold_signal(
        signal_df,
        date_col="Date",
        var_col="VaR_return_h",
        exposure_col="suggested_exposure",
        reduce_signal_col="reduce_signal",
        moderate_threshold_col="moderate_threshold",
        high_threshold_col="high_threshold",
        save_path=os.path.join(save_figures_dir, "post4_risk_threshold_signal.png"),
        show=show_plot,
    )

    return {
        "threshold_info": threshold_info,
        "signal_df": signal_df,
    }

# =========================================================
# 6) GARCH conditional volatility
# =========================================================

def build_portfolio_returns_from_csv(
    csv_path: str = "data/processed/log_returns.csv",
) -> pd.DataFrame:
    """
    로그수익률 CSV로부터 동일가중 포트폴리오 수익률 시계열 생성

    Parameters
    ----------
    csv_path : str
        종목별 로그수익률 CSV 경로

    Returns
    -------
    pd.DataFrame
        [Date, portfolio] 컬럼을 가진 데이터프레임
    """
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns:
        raise ValueError("CSV 파일에 'Date' 컬럼이 없습니다.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Date 제외한 수익률 컬럼만 사용
    asset_cols = [c for c in df.columns if c != "Date"]
    if len(asset_cols) == 0:
        raise ValueError("수익률 컬럼이 없습니다. Date 외 컬럼을 확인하세요.")

    returns_df = df[asset_cols].copy()

    # 결측치가 있는 행 제거
    valid_mask = returns_df.notna().all(axis=1)
    df = df.loc[valid_mask].copy()
    returns_df = returns_df.loc[valid_mask].copy()

    # 동일가중 포트폴리오
    weights = make_equal_weights(returns_df.columns.tolist())
    port = portfolio_return(returns_df, weights)

    out = pd.DataFrame({
        "Date": df["Date"].values,
        "portfolio": port.values
    })

    return out


def classify_vol_regime(
    vol_series: pd.Series,
    low_q: float = 0.33,
    high_q: float = 0.67,
) -> tuple[pd.Series, float, float]:
    """
    변동성 시계열을 분위수 기준으로 Low / Moderate / High regime으로 구분

    Parameters
    ----------
    vol_series : pd.Series
        GARCH 조건부 변동성 시계열
    low_q : float
        Low 구간 상단 분위수
    high_q : float
        High 구간 시작 분위수

    Returns
    -------
    regime : pd.Series
        low / moderate / high 라벨
    low_thr : float
        low 분위수 기준값
    high_thr : float
        high 분위수 기준값
    """
    low_thr = vol_series.quantile(low_q)
    high_thr = vol_series.quantile(high_q)

    regime = pd.Series(index=vol_series.index, dtype="object")
    regime[vol_series <= low_thr] = "low"
    regime[(vol_series > low_thr) & (vol_series <= high_thr)] = "moderate"
    regime[vol_series > high_thr] = "high"

    return regime, low_thr, high_thr


def compute_garch_volatility_regime(
    csv_path: str = "data/processed/log_returns.csv",
    out_csv_path: str = "results/tables/garch_volatility_regime.csv",
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> pd.DataFrame:
    """
    포트폴리오 수익률 기준 GARCH 조건부 변동성 및 volatility regime 계산

    Parameters
    ----------
    csv_path : str
        로그수익률 CSV 파일 경로
    out_csv_path : str
        결과 CSV 저장 경로
    p : int
        GARCH(p, q) 중 p 값
    q : int
        GARCH(p, q) 중 q 값
    dist : str
        오차분포 가정 ('normal', 't' 등)

    Returns
    -------
    pd.DataFrame
        Date, portfolio, garch_vol, regime, low_thr, high_thr
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    # 1. 포트폴리오 수익률 생성
    port_df = build_portfolio_returns_from_csv(csv_path=csv_path).copy()

    # 2. arch 패키지는 보통 % 단위 수익률에서 더 안정적으로 추정됨
    #    따라서 로그수익률(예: 0.01)을 100배 하여 사용
    ret_pct = port_df["portfolio"] * 100.0

    # 3. GARCH(1,1) 적합
    # mean='Zero': 평균을 0으로 두고 변동성 자체에 집중
    model = arch_model(
        ret_pct,
        mean="Zero",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        rescale=False
    )

    fitted = model.fit(disp="off")

    # 4. 조건부 변동성 추출
    # arch 결과도 % 단위이므로 다시 /100 하여 원래 수익률 스케일로 맞춤
    cond_vol = fitted.conditional_volatility / 100.0

    # 5. Regime 분류
    regime, low_thr, high_thr = classify_vol_regime(cond_vol)

    out = port_df.copy()
    out["garch_vol"] = cond_vol.values
    out["regime"] = regime.values
    out["low_thr"] = low_thr
    out["high_thr"] = high_thr

    out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv_path}")

    return out
# =========================================================
# 7) Runner
# =========================================================
def make_post4_input_from_port(port: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": port.index,
        "portfolio": port.values
    }).reset_index(drop=True)

def run_post4(
    df: pd.DataFrame,
    portfolio_value: float = 10_000_000,
    confidence_level: float = 0.95,
    lookback_days: int = 30,
    horizon_days: int = 5,
    risk_budget_pct: float = 0.08,
    moderate_ratio: float = 0.6,
    custom_scenarios: Optional[Dict[str, float]] = None,
    date_col: str = "Date",
    ret_col: str = "portfolio",
    save_tables_dir: str = "results/tables",
    save_figures_dir: str = "results/figures",
    show_plot: bool = True,
) -> Dict[str, Any]:
    """
    POST 4 전체 실행:
    - Stress Scenario table
    - VaR forecast summary (single point)
    - Risk decision (dynamic budget)
    - VaR forecast series
    - Realized horizon return
    - Plots + CSV 저장
    """
    os.makedirs(save_tables_dir, exist_ok=True)
    os.makedirs(save_figures_dir, exist_ok=True)

    # 1) Stress scenarios
    stress_df = run_stress_scenarios(
        df,
        portfolio_value=portfolio_value,
        ret_col=ret_col,
        scenarios=custom_scenarios,
        include_historical_worst=True,
    )

    # 2) Forecast summary
    forecast = forecast_var(
        df,
        confidence_level=confidence_level,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        portfolio_value=portfolio_value,
        ret_col=ret_col,
    )

    # 3) Decision
    decision = stop_loss_decision_dynamic(
        forecast,
        portfolio_value=portfolio_value,
        risk_budget_pct=risk_budget_pct,
        moderate_ratio=moderate_ratio,
    )

    # 4) Series
    var_series = build_var_forecast_series(
        df,
        confidence_level=confidence_level,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        portfolio_value=portfolio_value,
        date_col=date_col,
        ret_col=ret_col,
    )

    # 5) Realized
    realized_df = add_realized_horizon_return(
        df,
        horizon_days=horizon_days,
        date_col=date_col,
        ret_col=ret_col,
    )

    # 6) Plot: regime shading
    plot_var_with_regime_shading(
        var_series,
        risk_budget_pct=risk_budget_pct,
        moderate_ratio=moderate_ratio,
        date_col=date_col,
        var_col="VaR_return_h",
        save_path=os.path.join(save_figures_dir, "post4_var_regime_shading.png"),
        show=show_plot,
    )

    # 7) Plot: forecast vs realized (+ return merged df if needed)
    merged_df = plot_forecast_var_vs_realized_return(
        var_series,
        realized_df,
        date_col=date_col,
        var_col="VaR_return_h",
        realized_col="realized_return_h",
        save_path=os.path.join(save_figures_dir, "post4_forecast_vs_realized.png"),
        show=show_plot,
    )

    # 8) Forecast VaR 분포 + violation 기반 risk threshold 분석
    threshold_result = run_post4_risk_threshold_analysis(
        merged_df=merged_df,
        save_tables_dir=save_tables_dir,
        save_figures_dir=save_figures_dir,
        show_plot=show_plot,
    )

    # 9) Save CSVs
    stress_df.to_csv(os.path.join(save_tables_dir, "post4_stress_scenarios.csv"),
                     index=False, encoding="utf-8-sig")
    pd.DataFrame([forecast]).to_csv(os.path.join(save_tables_dir, "post4_var_forecast_summary.csv"),
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame([decision]).to_csv(os.path.join(save_tables_dir, "post4_risk_decision_summary.csv"),
                                    index=False, encoding="utf-8-sig")
    var_series.to_csv(os.path.join(save_tables_dir, "post4_var_forecast_series.csv"),
                      index=False, encoding="utf-8-sig")
    merged_df.to_csv(os.path.join(save_tables_dir, "post4_forecast_vs_realized_merged.csv"),
                     index=False, encoding="utf-8-sig")

    print("\n===== POST 4 완료 =====")
    print("Forecast:", forecast)
    print("Decision:", decision)

    return {
        "stress_df": stress_df,
        "forecast": forecast,
        "decision": decision,
        "var_series": var_series,
        "realized_df": realized_df,
        "merged_df": merged_df,
        "threshold_info": threshold_result["threshold_info"],
        "signal_df": threshold_result["signal_df"],
    }




