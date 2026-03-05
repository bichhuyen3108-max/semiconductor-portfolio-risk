import numpy as np
import pandas as pd
import yfinance as yf

from .config import TICKERS, START_DATE, END_DATE


def download_adj_close() -> pd.DataFrame:
    tickers = list(TICKERS.keys())

    df = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,      # Close is adjusted (split/dividend)
        progress=False,
        threads=False,         # avoid cache/db lock issues
        group_by="column"
    )

    prices = df["Close"].copy().sort_index()
    prices = prices[tickers]              # keep ticker order
    prices = prices.dropna(how="all")     # drop fully missing rows

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise RuntimeError(f"Missing tickers in download: {missing}")

    return prices

