import yfinance as yf
import pandas as pd

tickers = ["005930.KS", "000660.KS", "006400.KS"]

data = yf.download(tickers, start="2018-01-01", end="2024-01-01")["Adj Close"]

data.to_csv("data/semiconductor_prices.csv")

print("Data downloaded successfully.")
