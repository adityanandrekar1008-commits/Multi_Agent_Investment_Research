import yfinance as yf
import os


def get_balance_sheet(ticker):
    stock = yf.Ticker(ticker)

    balance_sheet = stock.balance_sheet

    return balance_sheet


if __name__ == "__main__":
    ticker = input("Enter company ticker: ")

    balance_sheet = get_balance_sheet(ticker)

    os.makedirs("data/raw", exist_ok=True)

    file_path = f"data/raw/{ticker}_balance_sheet.csv"
    balance_sheet.to_csv(file_path)

    print("\nBalance sheet saved successfully!")
    print("File:", file_path)

    print("\nBalance Sheet:")
    print(balance_sheet)