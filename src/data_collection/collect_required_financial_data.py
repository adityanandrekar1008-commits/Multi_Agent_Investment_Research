import yfinance as yf
import pandas as pd
import os
import time


# ---------------------------------------------------
# Companies
# ---------------------------------------------------

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSLA",
    "META",
    "JPM"
]


# ---------------------------------------------------
# Helper function
# ---------------------------------------------------

def get_metric(data, metric):

    if metric in data.index:
        return data.loc[metric]

    else:
        return pd.Series(
            index=data.columns,
            dtype="float64"
        )


# ---------------------------------------------------
# Collect data for one company
# ---------------------------------------------------

def collect_company_data(ticker):

    print("\n" + "=" * 60)
    print(f"Collecting required data for: {ticker}")
    print("=" * 60)

    stock = yf.Ticker(ticker)

    # ------------------------------------------------
    # Get financial statements
    # ------------------------------------------------

    income_data = stock.income_stmt
    balance_data = stock.balance_sheet

    if income_data.empty:
        print(f"No income statement found for {ticker}")
        return

    if balance_data.empty:
        print(f"No balance sheet found for {ticker}")
        return


    # ------------------------------------------------
    # Income statement data
    # ------------------------------------------------

    income_selected = pd.DataFrame({

        "Total Revenue": get_metric(
            income_data,
            "Total Revenue"
        ),

        "Operating Income": get_metric(
            income_data,
            "Operating Income"
        ),

        "EBIT": get_metric(
            income_data,
            "EBIT"
        ),

        "Net Income": get_metric(
            income_data,
            "Net Income"
        ),

        "Interest Expense": get_metric(
            income_data,
            "Interest Expense"
        )
    })


    # ------------------------------------------------
    # Balance sheet data
    # ------------------------------------------------

    balance_selected = pd.DataFrame({

        "Total Assets": get_metric(
            balance_data,
            "Total Assets"
        ),

        "Current Assets": get_metric(
            balance_data,
            "Current Assets"
        ),

        "Current Liabilities": get_metric(
            balance_data,
            "Current Liabilities"
        ),

        "Total Liabilities": get_metric(
            balance_data,
            "Total Liabilities"
        ),

        "Retained Earnings": get_metric(
            balance_data,
            "Retained Earnings"
        ),

        "Total Debt": get_metric(
            balance_data,
            "Total Debt"
        ),

        "Cash And Cash Equivalents": get_metric(
            balance_data,
            "Cash And Cash Equivalents"
        ),

        "Stockholders Equity": get_metric(
            balance_data,
            "Stockholders Equity"
        )
    })


    # ------------------------------------------------
    # Combine data
    # ------------------------------------------------

    combined = income_selected.join(
        balance_selected,
        how="outer"
    )


    # ------------------------------------------------
    # Add company identifier
    # ------------------------------------------------

    combined["Ticker"] = ticker


    # ------------------------------------------------
    # Convert index to Date
    # ------------------------------------------------

    combined.index = pd.to_datetime(
        combined.index
    )

    combined.index.name = "Date"


    # ------------------------------------------------
    # Reset index
    # ------------------------------------------------

    combined.reset_index(
        inplace=True
    )


    # ------------------------------------------------
    # Sort by date
    # ------------------------------------------------

    combined.sort_values(
        "Date",
        inplace=True
    )


    # ------------------------------------------------
    # Create output folder
    # ------------------------------------------------

    os.makedirs(
        "data/raw/required_financial_data",
        exist_ok=True
    )


    # ------------------------------------------------
    # Save company data
    # ------------------------------------------------

    output_file = (
        f"data/raw/required_financial_data/"
        f"{ticker}_required_financial_data.csv"
    )

    combined.to_csv(
        output_file,
        index=False
    )


    # ------------------------------------------------
    # Print result
    # ------------------------------------------------

    print("\nSaved successfully:")
    print(output_file)

    print("\nDataset Shape:")
    print(combined.shape)

    print("\nAvailable columns:")
    print(combined.columns.tolist())


# ---------------------------------------------------
# Main function
# ---------------------------------------------------

def main():

    for ticker in TICKERS:

        try:

            collect_company_data(ticker)

        except Exception as e:

            print(
                f"\nError collecting {ticker}: {e}"
            )

        # Small delay between requests
        time.sleep(2)


    print("\n" + "=" * 60)
    print("REQUIRED FINANCIAL DATA COLLECTION COMPLETED")
    print("=" * 60)


# ---------------------------------------------------
# Run program
# ---------------------------------------------------

if __name__ == "__main__":

    main()