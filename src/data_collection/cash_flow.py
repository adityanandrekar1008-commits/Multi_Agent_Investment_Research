import yfinance as yf
import os


def get_cash_flow(ticker):
    stock = yf.Ticker(ticker)

    cash_flow = stock.cashflow

    return cash_flow


if __name__ == "__main__":
    ticker = input("Enter company ticker: ")

    cash_flow = get_cash_flow(ticker)

    os.makedirs("data/raw", exist_ok=True)

    file_path = f"data/raw/{ticker}_cash_flow.csv"
    cash_flow.to_csv(file_path)

    print("\nCash flow data saved successfully!")
    print("File:", file_path)

    print("\nCash Flow:")
    print(cash_flow)