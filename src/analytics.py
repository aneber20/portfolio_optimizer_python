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
    
    return {'vol' : portfolio_vol}

# Price/Earnings Ratio
def calculate_portfolio_pe_ratio(tickers_and_holdings):
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
    
    return {'pe_ratio': portfolio_pe_ratio}

# 52 Week Return
def calculate_52_week_return(tickers_and_holdings):
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
    
    return {'52_week_return': portfolio_return}

# Sharpe Ratio
def calculate_sharpe_ratio(tickers_and_holdings, risk_free_rate=0.01):
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
    
    return {'sharpe_ratio': sharpe_ratio}


# RSI?

# Moving Average?