# Helper function to pull data from yfinance

import numpy as np
import pandas as pd
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    """Return a consistently formatted ticker symbol."""
    return ticker.strip().upper()


def set_holding_value(tickers_and_holdings: dict, ticker: str, amount: float) -> dict:
    """
    Add a holding or update an existing holding's dollar value.

    Parameters:
    tickers_and_holdings: dict
        Format: {'AAPL': 10000, 'MSFT': 5000, ...}
    ticker: str
        Stock ticker symbol.
    amount: float
        Dollar value of the holding. Must be greater than 0.
    """
    ticker = normalize_ticker(ticker)
    amount = float(amount)

    if not ticker:
        raise ValueError("Ticker is required")
    if amount <= 0:
        raise ValueError("Amount must be greater than 0")

    tickers_and_holdings[ticker] = amount
    return tickers_and_holdings


def remove_holding(tickers_and_holdings: dict, ticker: str) -> bool:
    """
    Remove a holding from the portfolio.

    Returns True if the ticker was removed, or False if it was not present.
    """
    ticker = normalize_ticker(ticker)
    if ticker in tickers_and_holdings:
        del tickers_and_holdings[ticker]
        return True
    return False


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
    data = fetch_price_history(tickers, period=f'{max_days}d')
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


def _to_price_frame(data) -> pd.DataFrame:
    """Normalize yfinance output into a Close-price dataframe."""
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        elif "Close" in data.columns.get_level_values(-1):
            prices = data.xs("Close", axis=1, level=-1)
        else:
            prices = data
    elif "Close" in data:
        prices = data[["Close"]]
    else:
        prices = data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna(how="all")


def fetch_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Fetch close prices for one or more tickers."""
    data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
    prices = _to_price_frame(data)
    if len(tickers) == 1:
        prices.columns = [tickers[0]]
    return prices


def calculate_portfolio_daily_returns(
    tickers_and_holdings: dict,
    period: str = "1y",
) -> pd.Series:
    """Calculate weighted daily portfolio returns."""
    tickers = list(tickers_and_holdings.keys())
    total_value = sum(tickers_and_holdings.values())
    prices = fetch_price_history(tickers, period)
    returns = prices.pct_change().dropna(how="all")

    weights = pd.Series(
        {ticker: tickers_and_holdings[ticker] / total_value for ticker in tickers}
    )
    aligned_returns = returns.reindex(columns=weights.index).dropna(how="all")
    return aligned_returns.mul(weights, axis=1).sum(axis=1).dropna()


def calculate_benchmark_daily_returns(
    benchmark_ticker: str = "VOO",
    period: str = "1y",
) -> pd.Series:
    """Calculate daily benchmark returns."""
    prices = fetch_price_history([benchmark_ticker], period)
    return prices.iloc[:, 0].pct_change().dropna()


def _annualized_return(daily_returns: pd.Series) -> float | None:
    if daily_returns.empty:
        return None
    cumulative_return = (1 + daily_returns).prod() - 1
    years = len(daily_returns) / 252
    if years <= 0:
        return None
    return (1 + cumulative_return) ** (1 / years) - 1


def _max_drawdown(daily_returns: pd.Series) -> float | None:
    if daily_returns.empty:
        return None
    growth = (1 + daily_returns).cumprod()
    drawdowns = growth / growth.cummax() - 1
    return float(drawdowns.min())


def _beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float,
) -> tuple[float | None, float | None]:
    joined = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if joined.empty or joined["benchmark"].var() == 0:
        return None, None

    beta = joined["portfolio"].cov(joined["benchmark"]) / joined["benchmark"].var()
    portfolio_ann = _annualized_return(joined["portfolio"])
    benchmark_ann = _annualized_return(joined["benchmark"])
    if portfolio_ann is None or benchmark_ann is None:
        return float(beta), None

    alpha = portfolio_ann - (risk_free_rate + beta * (benchmark_ann - risk_free_rate))
    return float(beta), float(alpha)


def calculate_performance_metrics(
    tickers_and_holdings: dict,
    period: str = "1y",
    benchmark_ticker: str = "VOO",
    risk_free_rate: float = 0.01,
) -> dict[str, float | None]:
    """Return simple-to-advanced performance metrics."""
    portfolio_returns = calculate_portfolio_daily_returns(tickers_and_holdings, period)
    benchmark_returns = calculate_benchmark_daily_returns(benchmark_ticker, period)
    beta, alpha = _beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate)

    cumulative_return = None
    if not portfolio_returns.empty:
        cumulative_return = float((1 + portfolio_returns).prod() - 1)

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": _annualized_return(portfolio_returns),
        "best_day": None if portfolio_returns.empty else float(portfolio_returns.max()),
        "worst_day": None if portfolio_returns.empty else float(portfolio_returns.min()),
        "win_rate": None if portfolio_returns.empty else float((portfolio_returns > 0).mean()),
        "alpha": alpha,
        "beta": beta,
    }


def calculate_risk_metrics(
    tickers_and_holdings: dict,
    period: str = "1y",
    benchmark_ticker: str = "VOO",
) -> dict[str, float | None]:
    """Return simple-to-advanced risk metrics."""
    portfolio_returns = calculate_portfolio_daily_returns(tickers_and_holdings, period)
    benchmark_returns = calculate_benchmark_daily_returns(benchmark_ticker, period)
    joined = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()

    downside_returns = portfolio_returns[portfolio_returns < 0]
    tracking_diff = joined["portfolio"] - joined["benchmark"] if not joined.empty else pd.Series(dtype=float)

    return {
        "volatility": None if portfolio_returns.empty else float(portfolio_returns.std() * np.sqrt(252)),
        "max_drawdown": _max_drawdown(portfolio_returns),
        "downside_deviation": None if downside_returns.empty else float(downside_returns.std() * np.sqrt(252)),
        "value_at_risk_95": None if portfolio_returns.empty else float(portfolio_returns.quantile(0.05)),
        "tracking_error": None if tracking_diff.empty else float(tracking_diff.std() * np.sqrt(252)),
        "correlation_to_benchmark": None if joined.empty else float(joined["portfolio"].corr(joined["benchmark"])),
    }


def calculate_portfolio_quality_metrics(tickers_and_holdings: dict) -> dict:
    """Return diversification and income-oriented portfolio quality metrics."""
    total_value = sum(tickers_and_holdings.values())
    weights = {
        ticker: amount / total_value for ticker, amount in tickers_and_holdings.items()
    }
    sector_weights = {}
    dividend_yields = []
    dividend_weights = []

    for ticker, weight in weights.items():
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector") or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0) + weight

        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None:
            dividend_yields.append(dividend_yield)
            dividend_weights.append(weight)

    return {
        "top_holding_weight": max(weights.values()) if weights else None,
        "hhi_concentration": sum(weight**2 for weight in weights.values()),
        "weighted_dividend_yield": (
            float(np.average(dividend_yields, weights=dividend_weights))
            if dividend_yields
            else None
        ),
        "sector_weights": sector_weights,
    }


def calculate_risk_adjusted_metrics(
    tickers_and_holdings: dict,
    period: str = "1y",
    benchmark_ticker: str = "VOO",
    risk_free_rate: float = 0.01,
) -> dict[str, float | None]:
    """Return advanced risk-adjusted portfolio metrics."""
    portfolio_returns = calculate_portfolio_daily_returns(tickers_and_holdings, period)
    benchmark_returns = calculate_benchmark_daily_returns(benchmark_ticker, period)
    joined = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()

    annualized_return = _annualized_return(portfolio_returns)
    max_drawdown = _max_drawdown(portfolio_returns)
    beta, _alpha = _beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate)

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = (
        None if downside_returns.empty else float(downside_returns.std() * np.sqrt(252))
    )

    volatility = None if portfolio_returns.empty else float(portfolio_returns.std() * np.sqrt(252))
    sharpe = None
    if annualized_return is not None and volatility and volatility != 0:
        sharpe = (annualized_return - risk_free_rate) / volatility

    sortino = None
    if annualized_return is not None and downside_deviation and downside_deviation != 0:
        sortino = (annualized_return - risk_free_rate) / downside_deviation

    tracking_diff = joined["portfolio"] - joined["benchmark"] if not joined.empty else pd.Series(dtype=float)
    tracking_error = None if tracking_diff.empty else float(tracking_diff.std() * np.sqrt(252))
    information_ratio = None
    if tracking_error and tracking_error != 0 and not joined.empty:
        active_return = (
            _annualized_return(joined["portfolio"])
            - _annualized_return(joined["benchmark"])
        )
        information_ratio = active_return / tracking_error

    treynor = None
    if annualized_return is not None and beta and beta != 0:
        treynor = (annualized_return - risk_free_rate) / beta

    calmar = None
    if annualized_return is not None and max_drawdown and max_drawdown != 0:
        calmar = annualized_return / abs(max_drawdown)

    upside_capture = None
    downside_capture = None
    if not joined.empty:
        upside = joined[joined["benchmark"] > 0]
        downside = joined[joined["benchmark"] < 0]
        if not upside.empty and upside["benchmark"].mean() != 0:
            upside_capture = upside["portfolio"].mean() / upside["benchmark"].mean()
        if not downside.empty and downside["benchmark"].mean() != 0:
            downside_capture = downside["portfolio"].mean() / downside["benchmark"].mean()

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "information_ratio": information_ratio,
        "treynor_ratio": treynor,
        "calmar_ratio": calmar,
        "upside_capture": None if upside_capture is None else float(upside_capture),
        "downside_capture": None if downside_capture is None else float(downside_capture),
    }
