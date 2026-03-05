import pandas as pd

from src.visualization import (
    plot_var_cvar_multi_levels,
    run_rolling_var_plots,
    run_backtesting_post3,
    load_portfolio_and_rolling_var,
    plot_return_vs_var_zoom,
    plot_var_models_only
)

from src.risk_framework import make_post4_input_from_port, run_post4


if __name__ == "__main__":

    # Post 1 ~ 3
    plot_var_cvar_multi_levels(conf_levels=(0.95, 0.99))
    run_rolling_var_plots()
    run_backtesting_post3()

    # Portfolio return load
    port, merged = load_portfolio_and_rolling_var()

    # Post4 input
    df_post4 = merged[["Date", "portfolio"]].copy()

    # Post4 실행
    run_post4(
        df=df_post4,
        portfolio_value=10_000_000,
        confidence_level=0.95,
        lookback_days=30,
        horizon_days=5,
        risk_budget_pct=0.08,
        show_plot=True,
    )

    hist_df = pd.read_csv(
    "results/tables/rolling_var.csv",
    parse_dates=["Date"]
    )

    models_df = pd.read_csv(
        "results/tables/rolling_var_models.csv",
        parse_dates=["Date"]
    )

    # Chart 1
    plot_return_vs_var_zoom(hist_df, models_df, 95)

    # Chart 2
    plot_var_models_only(hist_df, models_df, 95)
    