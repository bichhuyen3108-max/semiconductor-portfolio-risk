import pandas as pd

from src.visualization import (
    plot_var_cvar_multi_levels,
    run_rolling_var_plots,
    run_backtesting_post3,
    load_portfolio_and_rolling_var,
    plot_return_vs_var_zoom,
    plot_var_models_only,
    plot_garch_volatility_regime
)

from src.risk_framework import (make_input_from_port, run_post4,
                                 compute_garch_volatility_regime)


def run_garch_regime_analysis():
    """
    GARCH volatility regime 분석 실행
    """

    regime_df = compute_garch_volatility_regime(
        csv_path="data/processed/log_returns.csv",
        out_csv_path="results/tables/garch_volatility_regime.csv",
        p=1,
        q=1,
        dist="t",
    )

    plot_garch_volatility_regime(
        regime_df=regime_df,
        save_path="results/figures/garch_volatility_regime.png",
    )

if __name__ == "__main__":

    # =========================================================
    # Post 1 ~ 3 실행
    # =========================================================
    plot_var_cvar_multi_levels(conf_levels=(0.95, 0.99))
    run_rolling_var_plots()
    run_backtesting_post3()

    # =========================================================
    # 포트폴리오 수익률 데이터 불러오기
    # =========================================================
    port, merged = load_portfolio_and_rolling_var()

    # Post4 입력 데이터 구성
    df_post4 = merged[["Date", "portfolio"]].copy()

    # =========================================================
    # Post4 실행
    # - Stress Scenario
    # - Forecast VaR
    # - Realized Horizon Return
    # - Violation Analysis
    # - Risk Threshold Analysis
    # - 포트폴리오 비중 축소 신호 생성
    # =========================================================
    result = run_post4(
        df=df_post4,
        portfolio_value=10_000_000,
        confidence_level=0.95,
        lookback_days=30,
        horizon_days=5,
        risk_budget_pct=0.08,
        show_plot=True,
    )

    # =========================================================
    # Risk Threshold 결과 확인
    # =========================================================
    print("\n===== Risk Threshold Summary =====")
    print(result["threshold_info"])

    # =========================================================
    # 추가 비교 차트
    # =========================================================
    hist_df = pd.read_csv(
        "results/tables/rolling_var.csv",
        parse_dates=["Date"]
    )

    models_df = pd.read_csv(
        "results/tables/rolling_var_models.csv",
        parse_dates=["Date"]
    )

    # Chart 1: 실제 수익률 vs VaR 확대 구간 비교
    plot_return_vs_var_zoom(hist_df, models_df, 95)

    # Chart 2: VaR 모델 비교
    plot_var_models_only(hist_df, models_df, 95)
    
    # GARCH regime
    run_garch_regime_analysis()