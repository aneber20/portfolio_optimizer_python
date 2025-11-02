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
                <span style="font-size: 40px; font-weight: bold; color: #1f77b4;">{sp_returns[period]*100:.2f}%</span><br>
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

    # Display metrics with emphasis and explanations
    st.markdown(f"<h3 style='font-size: 24px;'>Volatility: {volatility:.2f}</h3>", unsafe_allow_html=True)
    st.markdown("**Volatility** measures the price fluctuations of your portfolio. Higher volatility indicates higher risk.")
    st.markdown("A high volatility (e.g., above 20%) may suggest that your portfolio is exposed to significant price swings, while low volatility (e.g., below 10%) indicates stability.")
    
    st.markdown(f"<h3 style='font-size: 24px;'>P/E Ratio: {pe_ratio:.2f}</h3>", unsafe_allow_html=True)
    st.markdown("**P/E Ratio** (Price-to-Earnings Ratio) helps assess if a stock is over or undervalued based on its earnings.")
    st.markdown("A high P/E ratio (e.g., above 25) may indicate overvaluation, while a low P/E ratio (e.g., below 15) could suggest undervaluation.")    
    
    st.markdown(f"<h3 style='font-size: 24px;'>Sharpe Ratio: {sharpe:.2f}</h3>", unsafe_allow_html=True)
    st.markdown("**Sharpe Ratio** measures risk-adjusted return, helping you understand how much excess return you receive for the extra volatility.")
    st.markdown("A Sharpe Ratio above 1 is generally considered good, while below 1 may indicate that the returns are not worth the risk.")
    
else:
    if not st.session_state.holdings:
        st.info("No holdings to analyze.")

# # Add a footnote at the bottom of the app
# st.markdown("---")
# st.markdown("Made by Adam Neber &middot; [Website](https://aneber20.github.io/)")