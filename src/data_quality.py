import numpy as np
import pandas as pd


def check_missing(prices: pd.DataFrame) -> pd.Series:
    missing_ratio = prices.isna().mean()

    print("Missing ratio per asset:")
    print(missing_ratio)

    return missing_ratio


def detect_outliers(returns: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
    # avoid divide-by-zero if any column has 0 std
    std = returns.std(ddof=0).replace(0, np.nan)
    z_scores = (returns - returns.mean()) / std
    z_scores = z_scores.replace([np.inf, -np.inf], np.nan)

    outliers = (np.abs(z_scores) > z_threshold)

    print("Extreme return observations (rows where any asset is outlier):")
    print(returns[outliers.any(axis=1)])

    return outliers


def check_trading_days(prices: pd.DataFrame) -> None:
    print("Date range:")
    print(prices.index.min(), "->", prices.index.max())

    print("Total observations:", len(prices))