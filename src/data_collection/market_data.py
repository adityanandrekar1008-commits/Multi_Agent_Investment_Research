import yfinance as yf


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)

    history = stock.history(period="1y")

    return history


if __name__ == "__main__":
    data = get_stock_data("AAPL")

    print("Last 5 days of stock data:")
    print(data.tail())