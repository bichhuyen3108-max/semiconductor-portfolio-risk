from src.visualization import plot_var_cvar_multi_levels
from src.visualization import run_rolling_var_plots
from src.visualization import run_backtesting_post3


if __name__ == "__main__":
    plot_var_cvar_multi_levels(conf_levels=(0.95, 0.99))
    run_rolling_var_plots()
    run_backtesting_post3()