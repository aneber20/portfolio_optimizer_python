from flask import Flask
import yfinance
import analytics

app = Flask(__name__)
@app.route('/')
def index():
    return """
    <h1>Welcome to the Stock Web App!</h1>
    <p>Use the following endpoints to interact with the API:</p>
    <ul>
        <li><strong>GET /add/&lt;ticker&gt;/&lt;amount&gt;</strong> - Add a ticker to the watchlist with the specified amount.</li>
        <li><strong>GET /watchlist</strong> - Retrieve the current watchlist.</li>
        <li><strong>GET /volatility</strong> - Calculate and retrieve the portfolio volatility.</li>
    </ul>
    """

# Store list of tickers, dollars held of that ticker
holdings = {}

# Add a ticker to the watchlist
@app.route('/add/<ticker>/<amount>')
def add_ticker(ticker, amount):
    # Convert ticker to uppercase for consistency
    ticker = ticker.upper()
    
    # Validate amount
    try:
        amount = float(amount)
        if amount <= 0:
            return "Error: Amount must be greater than 0"
    except ValueError:
        return "Error: Invalid amount format"
    
    # Verify if ticker exists using yfinance
    stock = yfinance.Ticker(ticker)
    info = stock.info
    if 'address1' not in info:
        return f"Error: {ticker} is not a valid stock ticker"
    
    # Add or update ticker and amount in holdings
    holdings[ticker] = amount
    return f"Added {ticker} with amount ${amount:,.2f} to holdings"
   
# Get the watchlist
@app.route('/watchlist')
def watchlist():
    return(holdings)

@app.route('/volatility')
def volatility():
    return(analytics.calculate_portfolio_volatility(holdings))

@app.route('/pe_ratio')
def pe_ratio():
    return(analytics.calculate_portfolio_pe_ratio(holdings))

@app.route('/52_week_return')
def return_52_week():
    return(analytics.calculate_52_week_return(holdings))

@app.route('/sharpe_ratio')
def sharpe_ratio():
    return(analytics.calculate_sharpe_ratio(holdings))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


