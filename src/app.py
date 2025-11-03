import streamlit as st
import yfinance as yf
from analytics import calculate_portfolio_volatility, calculate_portfolio_pe_ratio, calculate_52_week_return, calculate_sharpe_ratio, fetch_sp500_returns, fetch_portfolio_returns

# Initialize session state to store holdings if it doesn't exist
if 'holdings' not in st.session_state:
    st.session_state.holdings = {}

# Set the title of the Streamlit app
st.title('Portfolio Analyzer')

# Section to add a new stock to the portfolio
st.subheader('Add Stock to Portfolio')
col1, col2 = st.columns(2)

with col1:
    # Input for stock ticker
    ticker = st.text_input('Enter Stock Ticker', '').upper()
with col2:
    # Input for dollar amount
    amount = st.number_input('Enter Dollar Amount', min_value=0.0, value=0.0, step=100.0)

# Button to add stock to holdings
if st.button('Add Stock'):
    if ticker and amount > 0:
        try:
            # Verify if ticker exists using yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if hist.empty:
                # Show error if ticker is invalid
                st.error(f"No data found for ticker {ticker}")
            else:
                # Add valid ticker and amount to holdings
                st.session_state.holdings[ticker] = amount
                st.success(f"Added {ticker} with amount ${amount:,.2f} to holdings")
        except Exception as e:
            # Show error if yfinance fails
            st.error(f"Error: Unable to verify ticker {ticker}")
    else:
        # Show warnings for missing input
        if not ticker:
            st.warning("Please enter a ticker symbol")
        if not amount > 0:
            st.warning("Please enter an amount greater than 0")

# Display current holdings section
st.subheader('Current Holdings')
if st.session_state.holdings:
    # Create a dataframe for better display
    holdings_data = [[ticker, f"${amount:,.2f}"] for ticker, amount in st.session_state.holdings.items()]
    st.table({
        "Ticker": [row[0] for row in holdings_data],
        "Amount": [row[1] for row in holdings_data]
    })
else:
    # Info message if no holdings
    st.info("No holdings added yet")

# Button to finish adding holdings and show analytics
if st.button("Done Adding Holdings"):
    st.session_state.show_analytics = True

if st.session_state.get("show_analytics", False) and st.session_state.holdings:
    
    st.subheader("Portfolio Analytics")
    tickers = list(st.session_state.holdings.keys())
    amounts = list(st.session_state.holdings.values())
    
    # Calculate and display portfolio metrics
    volatility = calculate_portfolio_volatility(st.session_state.holdings)
    pe_ratio = calculate_portfolio_pe_ratio(st.session_state.holdings)
    sharpe = calculate_sharpe_ratio(st.session_state.holdings)

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
                <span style="font-size: 40px; font-weight: bold; color: #6f03fc;">{sp_returns[period]*100:.2f}%</span><br>
                <span style="font-size: 16px;">S&amp;P 500 Return</span>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 40px; font-weight: bold; color: #2ca02c;">{pf_returns[period]*100:.2f}%</span><br>
                <span style="font-size: 16px;">Portfolio Return</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display metrics with emphasis and flair
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: flex-start; gap: 40px; margin-bottom: 20px;">
            <div style="text-align: center;">
                <span style="font-size: 40px; font-weight: bold; color: #1f77b4;">{volatility:.2f}</span><br>
                <span style="font-size: 16px;">Volatility</span>
                <div style="font-size: 14px; color: #444;">Measures price fluctuations. Higher means more risk.<br>
                Less than 10: Stable | Greater than 20: Risky</div>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 40px; font-weight: bold; color: #1f77b4;">{pe_ratio:.2f}</span><br>
                <span style="font-size: 16px;">P/E Ratio</span>
                <div style="font-size: 14px; color: #444;">Assesses valuation based on earnings.<br>
                Less than 15: Undervalued | Greater than 20: Overvalued</div>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 40px; font-weight: bold; color: #1f77b4;">{sharpe:.2f}</span><br>
                <span style="font-size: 16px;">Sharpe Ratio</span>
                <div style="font-size: 14px; color: #444;">Risk-adjusted return. Above 1 is good.<br>
                Less than 1: Poor | 1-2: Good | Greater than 2: Excellent</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    if not st.session_state.holdings:
        st.info("No holdings to analyze.")

