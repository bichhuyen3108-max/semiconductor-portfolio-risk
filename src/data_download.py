import pandas as pd
import yfinance as yf
import os
from src.config import TICKERS, START_DATE, END_DATE


def download_adj_close() -> pd.DataFrame:
    tickers = list(TICKERS.keys())
    df = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    adj = df["Adj Close"].copy()
    adj = adj.sort_index()
    adj = adj[list(TICKERS.keys())] # TICKERS에서 정의한 순서를 유지하도록 컬럼 재정렬
                                    # (포트폴리오 가중치 계산 시 순서 불일치 방지)
    adj = adj.dropna(how="all")
    return adj

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    adj = download_adj_close()
    adj.to_csv("data/raw/adj_close.csv")
    print("Saved: data/raw/adj_close.csv", adj.shape)    