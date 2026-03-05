import numpy as np
import pandas as pd
from .config import TICKERS


def make_equal_weights(columns) -> pd.Series:
    n = len(columns)
    return pd.Series(1 / n, index=columns)

def portfolio_return(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    missing = set(returns.columns) - set(weights.index)
    extra = set(weights.index) - set(returns.columns)

    if missing:
        raise ValueError(f"weights에 없는 컬럼이 있습니다: {sorted(missing)}")
    if extra:
        raise ValueError(f"returns에 없는 ticker가 weights에 있습니다: {sorted(extra)}")

    if abs(weights.loc[returns.columns].sum() - 1.0) > 1e-6:
        raise ValueError("가중치 합이 1이 아닙니다. weights를 확인하세요.")

    aligned_weights = weights.loc[returns.columns]
    port = returns.dot(aligned_weights)
    port.name = "portfolio"
    return port

def normalize_weights(weights: dict[str, float] | np.ndarray,
                      tickers: list[str] | None = None) -> np.ndarray:
    """
    Convert weights into a numpy array aligned to tickers order, and normalize to sum=1.
    """
    if tickers is None:
        tickers = list(TICKERS.keys())

    if isinstance(weights, dict):
        w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    else:
        w = np.array(weights, dtype=float)

    if w.ndim != 1 or len(w) != len(tickers):
        raise ValueError(f"weights length must be {len(tickers)} (got {len(w)})")

    s = w.sum()
    if s == 0:
        raise ValueError("Sum of weights is 0. Please provide non-zero weights.")
    w = w / s
    return w


def compute_portfolio_returns(asset_returns: pd.DataFrame,
                              weights: dict[str, float] | np.ndarray,
                              tickers: list[str] | None = None) -> pd.Series:
    """
    Portfolio log return = sum_i (w_i * r_i) each day.
    asset_returns: DataFrame indexed by Date, columns = tickers
    """
    if tickers is None:
        tickers = list(TICKERS.keys())

    # ensure columns order
    missing = set(tickers) - set(asset_returns.columns)
    if missing:
        raise ValueError(f"asset_returns is missing columns: {sorted(missing)}")

    r = asset_returns[tickers].dropna(how="any")
    w = normalize_weights(weights, tickers)

    port_ret = r.to_numpy() @ w
    port_ret = pd.Series(port_ret, index=r.index, name="portfolio_log_return")
    return port_ret


def portfolio_cum_return(port_log_returns: pd.Series) -> pd.Series:
    """
    Convert log returns to cumulative return series: exp(cumsum(r)) - 1
    """
    cum = np.exp(port_log_returns.cumsum()) - 1.0
    cum.name = "portfolio_cum_return"
    return cum


def portfolio_value(port_log_returns: pd.Series, initial_value: float = 1_000_000.0) -> pd.Series:
    """
    Convert log returns to portfolio value curve.
    """
    value = initial_value * np.exp(port_log_returns.cumsum())
    value.name = "portfolio_value"
    return value