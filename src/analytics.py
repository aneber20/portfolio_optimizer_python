# Helper function to pull data from yfinance

import numpy as np
import pandas as pd
import yfinance as yf

# Volatility measure
def calculate_portfolio_volatility(tickers_and_holdings) -> float:
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
def calculate_portfolio_pe_ratio(tickers_and_holdings) -> float:
    """
    Calculate portfolio P/E ratio given a dictionary of tickers and their dollar holdings
    
    Parameters:
    tickers_and_holdings: dict
        Format: {'AAPL': 10000, 'MSFT': 5000, ...}
    """
    tickers = list(tickers_and_holdings.keys())
    total_value = sum(tickers_and_holdings.values())
    
    pe_ratios = []
    weights = []
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        pe_ratio = stock.info.get('forwardPE') or stock.info.get('trailingPE')
        
        if pe_ratio is not None:
            pe_ratios.append(pe_ratio)
            weights.append(tickers_and_holdings[ticker] / total_value)
    
    if not pe_ratios:
        return "Error: No valid P/E ratios found for the given tickers"
    
    portfolio_pe_ratio = np.average(pe_ratios, weights=weights)
    
    return portfolio_pe_ratio

# 52 Week Return
def calculate_52_week_return(tickers_and_holdings) -> float:
    """
    Calculate the 52-week return for the portfolio given a dictionary of tickers and their dollar holdings
    
    Parameters:
    tickers_and_holdings: dict
        Format: {'AAPL': 10000, 'MSFT': 5000, ...}
    """
    tickers = list(tickers_and_holdings.keys())
    total_value = sum(tickers_and_holdings.values())
    
    returns = []
    weights = []
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        
        if not hist.empty:
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            stock_return = (end_price - start_price) / start_price
            
            returns.append(stock_return)
            weights.append(tickers_and_holdings[ticker] / total_value)
    
    if not returns:
        return "Error: No valid returns found for the given tickers"
    
    portfolio_return = np.average(returns, weights=weights)
    
    return portfolio_return

# Sharpe Ratio
def calculate_sharpe_ratio(tickers_and_holdings, risk_free_rate=0.01) -> float:
    """
    Calculate the Sharpe Ratio for the portfolio given a dictionary of tickers and their dollar holdings
    
    Parameters:
    tickers_and_holdings: dict
        Format: {'AAPL': 10000, 'MSFT': 5000, ...}
    risk_free_rate: float
        The risk-free rate, default is 0.01 (1%)
    """
    tickers = list(tickers_and_holdings.keys())
    total_value = sum(tickers_and_holdings.values())
    
    # Download historical data
    data = yf.download(tickers, period='1y')['Close']
    
    # Calculate daily returns
    returns = data.pct_change()
    
    # Calculate expected portfolio return
    mean_returns = returns.mean()
    weights = np.array([tickers_and_holdings[ticker] / total_value for ticker in tickers])
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized
    
    # Calculate portfolio volatility
    cov_matrix = returns.cov() * 252  # Annualized
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    
    # Calculate Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return sharpe_ratio

# Helper function to get days from period string
# 252 days/y, 21/mo, 5/wk, 1/d
def _period_to_days(period: str) -> int:
   
    if period.endswith('y'):
        return int(period[:-1]) * 252
    elif period.endswith('mo'):
        return int(period[:-2]) * 21
    elif period.endswith('wk'):
        return int(period[:-2]) * 5
    elif period.endswith('d'):
        return int(period[:-1])
    else:
        return 0

# S&P 500 returns going back 10 years
def fetch_sp500_returns(periods: list[str]) -> dict[str, float]:
    
    ticker = "VOO"
    stock = yf.Ticker(ticker)
    
    returns = {}
    
    for p in periods:
        hist = stock.history(period = p)

        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        stock_return = (end_price - start_price) / start_price
        
        returns[p] = stock_return
        
    return returns


# Get portfolio returns for multiple periods using a single data pull.
# Assume in descending order
# Supported suffixes: 'y', 'mo', 'wk', 'd'
def fetch_portfolio_returns(periods: list[str], tickers_and_holdings: dict) -> dict[str, float]:
  
    tickers = list(tickers_and_holdings.keys())
    total_value = sum(tickers_and_holdings.values())
    
    # Calculate days for each period
    period_days = {p: _period_to_days(p) for p in periods}
    max_days = max(period_days.values())
    
    # Download historical data for all tickers for the longest period
    data = yf.download(tickers, period=f'{max_days}d')['Close']
    returns = {}
    for p in periods:
        days = period_days[p]
        if days and len(data) >= days:
            start_prices = data.iloc[-days]
            end_prices = data.iloc[-1]
            
            # Calculate weighted portfolio return for the period
            period_returns = []
            weights = []
            for ticker in tickers:
                if ticker in start_prices and ticker in end_prices:
                    r = (end_prices[ticker] - start_prices[ticker]) / start_prices[ticker]
                    period_returns.append(r)
                    weights.append(tickers_and_holdings[ticker] / total_value)
            if period_returns:
                returns[p] = float(np.average(period_returns, weights=weights))
            else:
                returns[p] = None
        else:
            returns[p] = None
    return returns

