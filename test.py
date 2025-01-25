import yfinance

data  = yfinance.download("AAPL", period="1y")
print(data.head())