import yfinance as yf
import os


def get_financial_data(ticker):
    stock = yf.Ticker(ticker)

    income_statement = stock.income_stmt

    return income_statement


if __name__ == "__main__":
    ticker = input("Enter company ticker: ")

    financial_data = get_financial_data(ticker)

    # Create raw data folder if it does not exist
    os.makedirs("data/raw", exist_ok=True)

    # Save financial data as CSV
    file_path = f"data/raw/{ticker}_financial_data.csv"
    financial_data.to_csv(file_path)

    print("\nFinancial data saved successfully!")
    print("File:", file_path)