import yfinance as yf
import pandas as pd

# KOSPI tickers
tickers = {
    "Samsung Electronics": "005930.KS",
    "SK Hynix": "000660.KS",
    "Samsung SDI": "006400.KS",
}

start_date = "2015-01-01"
end_date = "2024-12-31"

# Download (Close)
df = yf.download(list(tickers.values()), start=start_date, end=end_date)["Close"]
df.columns = list(tickers.keys())

# Save
df.to_csv("data/close_prices.csv", encoding="utf-8-sig")
print("✅ Saved: data/close_prices.csv")
print(df.tail())

