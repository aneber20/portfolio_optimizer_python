from flask import Flask
import yfinance
import analytics

app = Flask(__name__)
@app.route('/')
def index():
    return "Welcome to the Stock Web App!"

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

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


