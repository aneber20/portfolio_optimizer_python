from flask import Flask, render_template, jsonify, make_response
import yfinance
import analytics
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Add error handler for all exceptions
@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f"Error occurred: {str(error)}")
    return jsonify({
        "error": str(error),
        "status": 500
    }), 500

@app.route('/')
def index():
    return render_template('index.html')

# Store list of tickers, dollars held of that ticker
holdings = {}

# Add a ticker to the watchlist
@app.route('/add/<ticker>/<amount>')
def add_ticker(ticker, amount):
    app.logger.debug(f"Received request to add ticker={ticker}, amount={amount}")
    
    try:
        # Convert ticker to uppercase for consistency
        ticker = ticker.upper()
        app.logger.debug(f"Processing {ticker}")
        
        # Validate amount
        try:
            amount = float(amount)
            if amount <= 0:
                app.logger.debug(f"Invalid amount: {amount}")
                return jsonify({
                    "error": "Amount must be greater than 0",
                    "status": 400
                }), 400
        except ValueError as ve:
            app.logger.debug(f"Amount conversion error: {str(ve)}")
            return jsonify({
                "error": "Invalid amount format",
                "status": 400
            }), 400
        
        # Verify if ticker exists using yfinance
        app.logger.debug(f"Verifying ticker {ticker}")
        try:
            stock = yfinance.Ticker(ticker)
            # Get historical data - if ticker doesn't exist, this will fail
            hist = stock.history(period="1d")
            if hist.empty:
                app.logger.debug(f"No data found for ticker: {ticker}")
                return jsonify({
                    "error": f"No data found for ticker {ticker}",
                    "status": 404
                }), 404
        except Exception as ye:
            app.logger.debug(f"Error fetching ticker data: {str(ye)}")
            return jsonify({
                "error": f"Unable to verify ticker {ticker}",
                "status": 404
            }), 404
            
        # Add or update ticker and amount in holdings
        holdings[ticker] = amount
        response_data = {
            "message": f"Added {ticker} with amount ${amount:,.2f} to holdings",
            "status": 200
        }
        app.logger.debug(f"Success: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": 500
        }), 500

# Get the watchlist
@app.route('/watchlist')
def watchlist():
    return jsonify(holdings)

@app.route('/volatility')
def volatility():
    return jsonify(analytics.calculate_portfolio_volatility(holdings))

@app.route('/pe_ratio')
def pe_ratio():
    return jsonify(analytics.calculate_portfolio_pe_ratio(holdings))

@app.route('/52_week_return')
def return_52_week():
    return jsonify(analytics.calculate_52_week_return(holdings))

@app.route('/sharpe_ratio')
def sharpe_ratio():
    return jsonify(analytics.calculate_sharpe_ratio(holdings))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


