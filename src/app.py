import html

import streamlit as st
import yfinance as yf
import pandas as pd
from analytics import (
    calculate_benchmark_daily_returns,
    calculate_portfolio_daily_returns,
    calculate_performance_metrics,
    calculate_portfolio_pe_ratio,
    calculate_portfolio_quality_metrics,
    calculate_risk_adjusted_metrics,
    calculate_risk_metrics,
    fetch_price_history,
    fetch_sp500_returns,
    fetch_portfolio_returns,
    remove_holding,
    set_holding_value,
)

# Initialize session state to store holdings if it doesn't exist
if 'holdings' not in st.session_state:
    st.session_state.holdings = {}
if 'holding_units' not in st.session_state:
    st.session_state.holding_units = {}

# Set the title of the Streamlit app
st.title('GainSight')
st.header("At-a-glance stock portfolio analysis")


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_number(value):
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def growth_of_10000(returns):
    return (1 + returns).cumprod() * 10000


def drawdown_series(returns):
    growth = (1 + returns).cumprod()
    return growth / growth.cummax() - 1


def allocation_chart_data(holdings):
    total_value = sum(holdings.values())
    return pd.DataFrame(
        {
            "Ticker": holdings.keys(),
            "Weight": [amount / total_value * 100 for amount in holdings.values()],
            "Amount": holdings.values(),
        }
    ).set_index("Ticker")


def contribution_chart_data(holdings, period="1y"):
    tickers = list(holdings.keys())
    total_value = sum(holdings.values())
    prices = fetch_price_history(tickers, period)
    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        return pd.DataFrame()

    total_returns = (1 + returns).prod() - 1
    weights = pd.Series({ticker: holdings[ticker] / total_value for ticker in tickers})
    contributions = total_returns.reindex(weights.index).mul(weights).dropna() * 100
    return contributions.rename("Contribution").to_frame()


def rolling_metric_frame(portfolio_returns, benchmark_returns, risk_free_rate=0.01):
    joined = pd.concat(
        [portfolio_returns.rename("Portfolio"), benchmark_returns.rename("S&P 500")],
        axis=1,
    ).dropna()
    if len(joined) < 30:
        return pd.DataFrame()

    rolling_volatility = joined["Portfolio"].rolling(30).std() * (252**0.5)
    rolling_return = joined["Portfolio"].rolling(30).mean() * 252
    rolling_sharpe = (rolling_return - risk_free_rate) / rolling_volatility
    return pd.DataFrame(
        {
            "30D Volatility": rolling_volatility * 100,
            "30D Sharpe": rolling_sharpe,
        }
    ).dropna()


def get_latest_price(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


def holding_display_rows(holdings, holding_units):
    rows = []
    for ticker, amount in holdings.items():
        units = holding_units.get(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Shares": "N/A" if units is None else f"{units:,.4f}",
                "Amount": f"${amount:,.2f}",
            }
        )
    return rows


STATUS_STYLES = {
    "good": {"label": "Looks good", "color": "#15803d", "background": "#f0fdf4"},
    "neutral": {"label": "Neutral", "color": "#475569", "background": "#f8fafc"},
    "watch": {"label": "Worth reviewing", "color": "#b45309", "background": "#fffbeb"},
    "bad": {"label": "Needs attention", "color": "#b91c1c", "background": "#fef2f2"},
    "info": {"label": "Context only", "color": "#2563eb", "background": "#eff6ff"},
    "missing": {"label": "Not available", "color": "#64748b", "background": "#f8fafc"},
}


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value == "N/A" or value.startswith("Error:")
    return pd.isna(value)


def metric_status(metric_key, value):
    if is_missing(value):
        return "missing", "Could not calculate this from the available data."

    if metric_key in {"cumulative_return", "annualized_return", "alpha"}:
        if value > 0:
            return "good", "Above zero is generally positive."
        return "watch", "Below zero may deserve a closer look."

    if metric_key == "best_day":
        return "info", "Large gains are nice, but consistency matters more."

    if metric_key == "worst_day":
        if value <= -0.10:
            return "bad", "A single-day loss worse than 10% is severe."
        if value <= -0.05:
            return "watch", "A single-day loss worse than 5% is meaningful."
        return "good", "No unusually large one-day loss in this window."

    if metric_key == "win_rate":
        if value >= 0.55:
            return "good", "Above 55% means more up days than usual."
        if value >= 0.45:
            return "neutral", "Around 45%-55% is fairly normal."
        return "watch", "Below 45% means down days are more frequent."

    if metric_key == "beta":
        if 0.8 <= value <= 1.2:
            return "good", "Around 1.0 means market-like sensitivity."
        if value > 1.5:
            return "watch", "Above 1.5 means high market sensitivity."
        if value < 0.5:
            return "info", "Below 0.5 means low market sensitivity."
        return "neutral", "Somewhat different from the market."

    if metric_key in {"volatility", "downside_deviation", "tracking_error"}:
        if value < 0.15:
            return "good", "Below 15% is relatively calm."
        if value <= 0.30:
            return "watch", "15%-30% is a moderate risk range."
        return "bad", "Above 30% is high and worth investigating."

    if metric_key == "max_drawdown":
        if value <= -0.30:
            return "bad", "A drawdown worse than 30% is severe."
        if value <= -0.15:
            return "watch", "A drawdown worse than 15% is meaningful."
        return "good", "Drawdown has stayed below 15% in this window."

    if metric_key == "value_at_risk_95":
        if value <= -0.05:
            return "bad", "A 5% daily VaR loss is very large."
        if value <= -0.025:
            return "watch", "A 2.5%-5% daily VaR loss is notable."
        return "good", "Daily VaR is below 2.5%."

    if metric_key == "correlation_to_benchmark":
        if value >= 0.85:
            return "watch", "Highly benchmark-like; diversification may be limited."
        if value >= 0.5:
            return "neutral", "Moderate market relationship."
        return "good", "Lower correlation can help diversification."

    if metric_key == "pe_ratio":
        if isinstance(value, str):
            return "missing", value
        if value < 0:
            return "watch", "Negative P/E often reflects negative earnings."
        if value < 15:
            return "good", "Below 15 can suggest cheaper valuation."
        if value <= 25:
            return "neutral", "15-25 is a common broad-market range."
        return "watch", "Above 25 may imply richer valuation."

    if metric_key == "top_holding_weight":
        if value <= 0.15:
            return "good", "Below 15% is reasonably diversified."
        if value <= 0.30:
            return "watch", "15%-30% means one holding matters a lot."
        return "bad", "Above 30% is concentrated."

    if metric_key == "weighted_dividend_yield":
        if value < 0.01:
            return "info", "Low yield; likely more growth-oriented."
        if value <= 0.04:
            return "good", "1%-4% is a common income range."
        return "watch", "Very high yield can signal elevated risk."

    if metric_key == "hhi_concentration":
        if value <= 0.10:
            return "good", "Below 0.10 suggests broad diversification."
        if value <= 0.25:
            return "watch", "0.10-0.25 suggests moderate concentration."
        return "bad", "Above 0.25 suggests high concentration."

    if metric_key in {"sharpe_ratio", "sortino_ratio", "information_ratio"}:
        if value >= 1:
            return "good", "Above 1.0 is generally strong."
        if value >= 0:
            return "watch", "Between 0 and 1 is positive but not strong."
        return "bad", "Below 0 means risk was not rewarded."

    if metric_key == "treynor_ratio":
        if value > 0:
            return "good", "Above zero means positive excess return per beta."
        return "bad", "Below zero means beta risk was not rewarded."

    if metric_key == "calmar_ratio":
        if value >= 1:
            return "good", "Above 1.0 means return exceeded max drawdown."
        if value >= 0.5:
            return "watch", "0.5-1.0 is okay but not especially strong."
        return "bad", "Below 0.5 suggests drawdowns dominate returns."

    if metric_key == "upside_capture":
        if value >= 1:
            return "good", "Above 100% means beating the benchmark on up days."
        if value >= 0.8:
            return "neutral", "80%-100% captures most benchmark upside."
        return "watch", "Below 80% may lag in rising markets."

    if metric_key == "downside_capture":
        if value <= 0.8:
            return "good", "Below 80% means less downside than the benchmark."
        if value <= 1:
            return "neutral", "80%-100% is near benchmark downside."
        return "watch", "Above 100% means falling more than the benchmark."

    return "info", "Use this alongside the other metrics."


def metric_grid(metrics):
    cols = st.columns(3)
    for index, (label, raw_value, display_value, help_text, metric_key) in enumerate(metrics):
        status_key, baseline_text = metric_status(metric_key, raw_value)
        style = STATUS_STYLES[status_key]
        with cols[index % 3]:
            st.markdown(
                f"""
                <div style="border: 1px solid #e2e8f0; border-left: 5px solid {style["color"]}; border-radius: 8px; padding: 14px 14px 12px; margin-bottom: 12px; background: {style["background"]}; min-height: 150px;">
                    <div style="font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 6px;">{html.escape(label)}</div>
                    <div style="font-size: 30px; line-height: 1.15; font-weight: 750; color: {style["color"]}; margin-bottom: 8px;">{html.escape(display_value)}</div>
                    <div style="font-size: 12px; font-weight: 700; color: {style["color"]}; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">{style["label"]}</div>
                    <div style="font-size: 13px; color: #475569; line-height: 1.35;">{html.escape(help_text)} {html.escape(baseline_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Section to add a new stock to the portfolio
st.subheader('Add Stock to Portfolio')
entry_mode = st.radio(
    "Add by",
    ["Dollar amount", "Shares owned"],
    horizontal=True,
)
col1, col2 = st.columns(2)

with col1:
    # Input for stock ticker
    ticker = st.text_input('Enter Stock Ticker', '').strip().upper()
with col2:
    if entry_mode == "Dollar amount":
        amount = st.number_input(
            'Enter Dollar Amount',
            min_value=0.0,
            value=0.0,
            step=100.0,
        )
        shares = None
    else:
        shares = st.number_input(
            'Enter Shares Owned',
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.4f",
        )
        amount = 0.0

if entry_mode == "Shares owned":
    st.caption("Shares are converted to dollar value using the latest available close price.")

# Button to add stock to holdings
if st.button('Add Stock'):
    has_valid_input = amount > 0 if entry_mode == "Dollar amount" else shares > 0
    if ticker and has_valid_input:
        try:
            # Verify if ticker exists using yfinance
            latest_price = get_latest_price(ticker)
            
            if latest_price is None:
                # Show error if ticker is invalid
                st.error(f"No data found for ticker {ticker}")
            else:
                if entry_mode == "Shares owned":
                    amount = shares * latest_price
                else:
                    st.session_state.holding_units.pop(ticker, None)

                # Add valid ticker and amount to holdings
                set_holding_value(st.session_state.holdings, ticker, amount)
                if entry_mode == "Shares owned":
                    st.session_state.holding_units[ticker] = shares
                    st.success(
                        f"Added {shares:,.4f} shares of {ticker} "
                        f"at about ${latest_price:,.2f}/share "
                        f"(${amount:,.2f}) to holdings"
                    )
                else:
                    st.success(f"Added {ticker} with amount ${amount:,.2f} to holdings")
        except Exception as e:
            # Show error if yfinance fails
            st.error(f"Error: Unable to verify ticker {ticker}")
    else:
        # Show warnings for missing input
        if not ticker:
            st.warning("Please enter a ticker symbol")
        if entry_mode == "Dollar amount" and not amount > 0:
            st.warning("Please enter an amount greater than 0")
        if entry_mode == "Shares owned" and not shares > 0:
            st.warning("Please enter shares greater than 0")

# Display current holdings section
st.subheader('Current Holdings')
if st.session_state.holdings:
    # Create a dataframe for better display
    holdings_data = holding_display_rows(
        st.session_state.holdings,
        st.session_state.holding_units,
    )
    st.table(holdings_data)

    st.subheader("Manage Holdings")
    selected_ticker = st.selectbox("Select Holding", list(st.session_state.holdings.keys()))
    update_mode = st.radio(
        "Update by",
        ["Dollar amount", "Shares owned"],
        horizontal=True,
        key="update_mode",
    )
    if update_mode == "Dollar amount":
        updated_amount = st.number_input(
            "New Dollar Amount",
            min_value=0.0,
            value=float(st.session_state.holdings[selected_ticker]),
            step=100.0,
        )
        updated_shares = None
    else:
        updated_shares = st.number_input(
            "New Shares Owned",
            min_value=0.0,
            value=float(st.session_state.holding_units.get(selected_ticker, 0.0)),
            step=1.0,
            format="%.4f",
        )
        updated_amount = 0.0
        st.caption("Shares are converted to dollar value using the latest available close price.")
    manage_col1, manage_col2 = st.columns(2)

    with manage_col1:
        if st.button("Update Holding"):
            try:
                if update_mode == "Shares owned":
                    latest_price = get_latest_price(selected_ticker)
                    if latest_price is None:
                        st.error(f"No price data found for {selected_ticker}")
                    else:
                        updated_amount = updated_shares * latest_price
                else:
                    st.session_state.holding_units.pop(selected_ticker, None)

                set_holding_value(st.session_state.holdings, selected_ticker, updated_amount)
                if update_mode == "Shares owned":
                    st.session_state.holding_units[selected_ticker] = updated_shares
                    st.success(
                        f"Updated {selected_ticker} to {updated_shares:,.4f} shares "
                        f"(${updated_amount:,.2f})"
                    )
                else:
                    st.success(f"Updated {selected_ticker} to ${updated_amount:,.2f}")
            except ValueError as e:
                st.warning(str(e))

    with manage_col2:
        if st.button("Remove Holding"):
            if remove_holding(st.session_state.holdings, selected_ticker):
                st.session_state.holding_units.pop(selected_ticker, None)
                st.success(f"Removed {selected_ticker} from holdings")
                st.rerun()
            else:
                st.warning(f"{selected_ticker} is not in holdings")
else:
    # Info message if no holdings
    st.info("No holdings added yet")

# Button to finish adding holdings and show analytics
if st.button("Done Adding Holdings"):
    st.session_state.show_analytics = True

if st.session_state.get("show_analytics", False) and st.session_state.holdings:
    
    
    tickers = list(st.session_state.holdings.keys())
    amounts = list(st.session_state.holdings.values())
    
    # Calculate and display portfolio metrics
    pe_ratio = calculate_portfolio_pe_ratio(st.session_state.holdings)

    # Calculate S&P 500 returns over the past 10, 5, 1 years, 6, 3, 1 months, 2 weeks
    periods = ['10y', '5y', '1y', '6mo', '3mo', '1mo', '1wk']
    sp_returns = fetch_sp500_returns(periods)
    pf_returns = fetch_portfolio_returns(periods, st.session_state.holdings)
    
    st.subheader("Compare Portfolio Returns with S&P 500")
    # Row of buttons for period selection
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = periods[0]
    cols = st.columns(len(periods))
    for i, p in enumerate(periods):
        if cols[i].button(p):
            st.session_state.selected_period = p

    period = st.session_state.selected_period
    
    # Display the comparison result
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-bottom: 20px;">
            <div style="text-align: center;">
                <span style="font-size: 40px; font-weight: bold; color: #6f03fc;">{format_percent(sp_returns.get(period))}</span><br>
                <span style="font-size: 16px;">S&amp;P 500 Return</span>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 40px; font-weight: bold; color: #2ca02c;">{format_percent(pf_returns.get(period))}</span><br>
                <span style="font-size: 16px;">Portfolio Return</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Portfolio Analytics")

    performance_metrics = calculate_performance_metrics(st.session_state.holdings)
    risk_metrics = calculate_risk_metrics(st.session_state.holdings)
    quality_metrics = calculate_portfolio_quality_metrics(st.session_state.holdings)
    risk_adjusted_metrics = calculate_risk_adjusted_metrics(st.session_state.holdings)
    portfolio_daily_returns = calculate_portfolio_daily_returns(st.session_state.holdings)
    benchmark_daily_returns = calculate_benchmark_daily_returns()

    performance_tab, risk_tab, quality_tab, advanced_tab = st.tabs(
        ["Performance", "Risk", "Portfolio Quality", "Risk-Adjusted"]
    )

    with performance_tab:
        growth_data = pd.concat(
            [
                growth_of_10000(portfolio_daily_returns).rename("Portfolio"),
                growth_of_10000(benchmark_daily_returns).rename("S&P 500"),
            ],
            axis=1,
        ).dropna()
        if not growth_data.empty:
            st.caption("Growth of $10,000")
            st.line_chart(growth_data)

        period_return_data = pd.DataFrame(
            {
                "Portfolio": {
                    period_label: return_value * 100
                    for period_label, return_value in pf_returns.items()
                    if return_value is not None
                },
                "S&P 500": {
                    period_label: return_value * 100
                    for period_label, return_value in sp_returns.items()
                    if return_value is not None
                },
            }
        )
        if not period_return_data.empty:
            st.caption("Return by Period")
            st.bar_chart(period_return_data)

        contribution_data = contribution_chart_data(st.session_state.holdings)
        if not contribution_data.empty:
            st.caption("Contribution to 1Y Return by Holding")
            st.bar_chart(contribution_data)

        metric_grid(
            [
                (
                    "Cumulative Return",
                    performance_metrics["cumulative_return"],
                    format_percent(performance_metrics["cumulative_return"]),
                    "Total portfolio return over the selected history window.",
                    "cumulative_return",
                ),
                (
                    "Annualized Return",
                    performance_metrics["annualized_return"],
                    format_percent(performance_metrics["annualized_return"]),
                    "Portfolio return converted into a yearly rate.",
                    "annualized_return",
                ),
                (
                    "Best Day",
                    performance_metrics["best_day"],
                    format_percent(performance_metrics["best_day"]),
                    "Largest single-day portfolio gain.",
                    "best_day",
                ),
                (
                    "Worst Day",
                    performance_metrics["worst_day"],
                    format_percent(performance_metrics["worst_day"]),
                    "Largest single-day portfolio loss.",
                    "worst_day",
                ),
                (
                    "Win Rate",
                    performance_metrics["win_rate"],
                    format_percent(performance_metrics["win_rate"]),
                    "Share of trading days with a positive portfolio return.",
                    "win_rate",
                ),
                (
                    "Alpha vs S&P 500",
                    performance_metrics["alpha"],
                    format_percent(performance_metrics["alpha"]),
                    "Return above what would be expected after adjusting for market exposure.",
                    "alpha",
                ),
                (
                    "Beta vs S&P 500",
                    performance_metrics["beta"],
                    format_number(performance_metrics["beta"]),
                    "Sensitivity to S&P 500 movement. 1.00 means roughly market-like.",
                    "beta",
                ),
            ]
        )

    with risk_tab:
        drawdown_data = pd.concat(
            [
                (drawdown_series(portfolio_daily_returns) * 100).rename("Portfolio"),
                (drawdown_series(benchmark_daily_returns) * 100).rename("S&P 500"),
            ],
            axis=1,
        ).dropna()
        if not drawdown_data.empty:
            st.caption("Drawdown from Prior Peak")
            st.line_chart(drawdown_data)

        rolling_risk_data = rolling_metric_frame(
            portfolio_daily_returns,
            benchmark_daily_returns,
        )
        if not rolling_risk_data.empty:
            st.caption("Rolling 30-Day Portfolio Volatility")
            st.line_chart(rolling_risk_data[["30D Volatility"]])

        metric_grid(
            [
                (
                    "Volatility",
                    risk_metrics["volatility"],
                    format_percent(risk_metrics["volatility"]),
                    "Annualized fluctuation in daily returns.",
                    "volatility",
                ),
                (
                    "Max Drawdown",
                    risk_metrics["max_drawdown"],
                    format_percent(risk_metrics["max_drawdown"]),
                    "Largest peak-to-trough decline over the history window.",
                    "max_drawdown",
                ),
                (
                    "Downside Deviation",
                    risk_metrics["downside_deviation"],
                    format_percent(risk_metrics["downside_deviation"]),
                    "Annualized volatility using only negative return days.",
                    "downside_deviation",
                ),
                (
                    "Daily VaR 95%",
                    risk_metrics["value_at_risk_95"],
                    format_percent(risk_metrics["value_at_risk_95"]),
                    "Estimated daily loss threshold exceeded on about 5% of days.",
                    "value_at_risk_95",
                ),
                (
                    "Tracking Error",
                    risk_metrics["tracking_error"],
                    format_percent(risk_metrics["tracking_error"]),
                    "Annualized volatility of returns relative to the S&P 500.",
                    "tracking_error",
                ),
                (
                    "Correlation to S&P 500",
                    risk_metrics["correlation_to_benchmark"],
                    format_number(risk_metrics["correlation_to_benchmark"]),
                    "How closely the portfolio moves with the S&P 500.",
                    "correlation_to_benchmark",
                ),
            ]
        )

    with quality_tab:
        allocation_data = allocation_chart_data(st.session_state.holdings)
        if not allocation_data.empty:
            st.caption("Current Allocation by Holding")
            st.bar_chart(allocation_data[["Weight"]])

        metric_grid(
            [
                (
                    "P/E Ratio",
                    pe_ratio,
                    format_number(pe_ratio),
                    "Weighted valuation based on forward or trailing earnings.",
                    "pe_ratio",
                ),
                (
                    "Top Holding Weight",
                    quality_metrics["top_holding_weight"],
                    format_percent(quality_metrics["top_holding_weight"]),
                    "The largest single position as a share of the portfolio.",
                    "top_holding_weight",
                ),
                (
                    "Dividend Yield",
                    quality_metrics["weighted_dividend_yield"],
                    format_percent(quality_metrics["weighted_dividend_yield"]),
                    "Weighted dividend yield for holdings with available yield data.",
                    "weighted_dividend_yield",
                ),
                (
                    "HHI Concentration",
                    quality_metrics["hhi_concentration"],
                    format_number(quality_metrics["hhi_concentration"]),
                    "Diversification score. Higher means the portfolio is more concentrated.",
                    "hhi_concentration",
                ),
            ]
        )

        sector_weights = quality_metrics["sector_weights"]
        if sector_weights:
            st.caption("Sector Concentration")
            sector_chart_data = pd.DataFrame(
                {
                    "Sector": sector_weights.keys(),
                    "Weight": [value * 100 for value in sector_weights.values()],
                }
            ).set_index("Sector")
            st.bar_chart(sector_chart_data)

    with advanced_tab:
        rolling_advanced_data = rolling_metric_frame(
            portfolio_daily_returns,
            benchmark_daily_returns,
        )
        if not rolling_advanced_data.empty:
            st.caption("Rolling 30-Day Sharpe Ratio")
            st.line_chart(rolling_advanced_data[["30D Sharpe"]])

        metric_grid(
            [
                (
                    "Sharpe Ratio",
                    risk_adjusted_metrics["sharpe_ratio"],
                    format_number(risk_adjusted_metrics["sharpe_ratio"]),
                    "Excess return per unit of total volatility.",
                    "sharpe_ratio",
                ),
                (
                    "Sortino Ratio",
                    risk_adjusted_metrics["sortino_ratio"],
                    format_number(risk_adjusted_metrics["sortino_ratio"]),
                    "Excess return per unit of downside volatility.",
                    "sortino_ratio",
                ),
                (
                    "Information Ratio",
                    risk_adjusted_metrics["information_ratio"],
                    format_number(risk_adjusted_metrics["information_ratio"]),
                    "Excess return versus the benchmark per unit of tracking error.",
                    "information_ratio",
                ),
                (
                    "Treynor Ratio",
                    risk_adjusted_metrics["treynor_ratio"],
                    format_number(risk_adjusted_metrics["treynor_ratio"]),
                    "Excess return per unit of market beta.",
                    "treynor_ratio",
                ),
                (
                    "Calmar Ratio",
                    risk_adjusted_metrics["calmar_ratio"],
                    format_number(risk_adjusted_metrics["calmar_ratio"]),
                    "Annualized return divided by maximum drawdown.",
                    "calmar_ratio",
                ),
                (
                    "Upside Capture",
                    risk_adjusted_metrics["upside_capture"],
                    format_percent(risk_adjusted_metrics["upside_capture"]),
                    "Average portfolio participation when the S&P 500 is up.",
                    "upside_capture",
                ),
                (
                    "Downside Capture",
                    risk_adjusted_metrics["downside_capture"],
                    format_percent(risk_adjusted_metrics["downside_capture"]),
                    "Average portfolio participation when the S&P 500 is down.",
                    "downside_capture",
                ),
            ]
        )
else:
    if not st.session_state.holdings:
        st.info("No holdings to analyze.")
