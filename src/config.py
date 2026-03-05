from datetime import datetime, timedelta

TICKERS = {
    "005930.KS": "Samsung Electronics",
    "000660.KS": "SK Hynix",
    "006400.KS": "Samsung SDI",
}

START_DATE = "2016-01-01"
END_DATE = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")  
CONF_LEVELS = [0.95, 0.99]
TRADING_DAYS = 252