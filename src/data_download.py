import os
import pandas as pd
import yfinance as yf

from .config import TICKERS, START_DATE, END_DATE


def download_adj_close() -> pd.DataFrame:
    # 티커 목록 추출
    tickers = list(TICKERS.keys())

    # Yahoo Finance에서 수정주가 다운로드
    df = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=None,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column"
    )

    # 다운로드 결과 확인
    if df.empty:
        raise RuntimeError("다운로드 결과가 비어 있습니다.")

    # Close 컬럼 추출
    if "Close" not in df.columns.get_level_values(0):
        raise RuntimeError(f"'Close' 컬럼을 찾을 수 없습니다. columns={df.columns}")

    prices = df["Close"].copy().sort_index()

    # 티커 순서 유지
    prices = prices.reindex(columns=tickers)

    # 모든 값이 결측치인 행 제거
    prices = prices.dropna(how="all")

    # 누락 티커 확인
    missing = set(tickers) - set(prices.columns[prices.notna().any()])
    if missing:
        raise RuntimeError(f"다음 티커 데이터가 누락되었습니다: {missing}")

    return prices

def main():
    # 데이터 다운로드
    prices = download_adj_close()

    # 저장 폴더 생성
    os.makedirs("data/raw", exist_ok=True)

    # CSV 저장
    save_path = "data/raw/adj_close.csv"
    prices.to_csv(save_path)

    print(f"Saved: {save_path}")
    print(prices.head())



if __name__ == "__main__":
    main()
