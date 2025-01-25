# Helper function to pull data from yfinance

import numpy as np
import pandas as pd
import yfinance as yf

# Volatility measure
def calculate_portfolio_volatility(tickers_and_holdings):
    """
    Calculate portfolio volatility given a dictionary of tickers and their dollar holdings
    
    Parameters:
    tickers_and_holdings: dict
        Format: {'AAPL': 10000, 'MSFT': 5000, ...}
    """
    # Download historical data
    tickers = list(tickers_and_holdings.keys())
    data = yf.download(tickers, period='1y')['Close']
    
    # Calculate daily returns
    returns = data.pct_change()
    
    # Calculate covariance matrix
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Calculate weights (proportion of each holding)
    total_value = sum(tickers_and_holdings.values())
    weights = np.array([tickers_and_holdings[ticker]/total_value for ticker in tickers])
    
    # Calculate portfolio volatility
    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    
    return portfolio_vol

# Price/Earnings Ratio

# 52 Week Return

# Sharpe Ratio

# RSI?

# Moving Average?