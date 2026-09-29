import yfinance as yf
import pandas as pd
import os
import time


# Companies for the initial dataset
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


def get_metric(data, metric):
    if metric in data.index:
        return data.loc[metric]
    else:
        return pd.Series(index=data.columns, dtype="float64")


def collect_company_data(ticker):

    print(f"\n{'=' * 50}")
    print(f"Collecting data for: {ticker}")
    print(f"{'=' * 50}")

    stock = yf.Ticker(ticker)

    # Get financial statements
    income_data = stock.income_stmt
    balance_data = stock.balance_sheet
    cashflow_data = stock.cashflow

    if income_data.empty:
        print(f"No income statement data found for {ticker}")
        return

    # ---------------------------------------------------
    # Select financial metrics
    # ---------------------------------------------------

    data = pd.DataFrame({
        "Total Revenue": get_metric(
            income_data, "Total Revenue"
        ),
        "Gross Profit": get_metric(
            income_data, "Gross Profit"
        ),
        "Operating Income": get_metric(
            income_data, "Operating Income"
        ),
        "Net Income": get_metric(
            income_data, "Net Income"
        ),
        "EBITDA": get_metric(
            income_data, "EBITDA"
        ),
        "Operating Expense": get_metric(
            income_data, "Operating Expense"
        ),
        "Cost Of Revenue": get_metric(
            income_data, "Cost Of Revenue"
        ),
        "Research And Development": get_metric(
            income_data, "Research And Development"
        ),
        "Interest Expense": get_metric(
            income_data, "Interest Expense"
        ),
        "Total Debt": get_metric(
            balance_data, "Total Debt"
        ),
        "Cash And Cash Equivalents": get_metric(
            balance_data, "Cash And Cash Equivalents"
        ),
        "Tangible Book Value": get_metric(
            balance_data, "Tangible Book Value"
        ),
        "Free Cash Flow": get_metric(
            cashflow_data, "Free Cash Flow"
        ),
        "Operating Cash Flow": get_metric(
            cashflow_data, "Operating Cash Flow"
        ),
        "Capital Expenditure": get_metric(
            cashflow_data, "Capital Expenditure"
        )
    })

    # ---------------------------------------------------
    # Financial ratios
    # ---------------------------------------------------

    data["Profit Margin"] = (
        data["Net Income"] /
        data["Total Revenue"]
    )

    data["Operating Margin"] = (
        data["Operating Income"] /
        data["Total Revenue"]
    )

    data["EBITDA Margin"] = (
        data["EBITDA"] /
        data["Total Revenue"]
    )

    data["Gross Profit Margin"] = (
        data["Gross Profit"] /
        data["Total Revenue"]
    )

    data["Debt to Cash"] = (
        data["Total Debt"] /
        data["Cash And Cash Equivalents"]
    )

    data["Free Cash Flow Margin"] = (
        data["Free Cash Flow"] /
        data["Total Revenue"]
    )

    data["Operating Cash Flow Margin"] = (
        data["Operating Cash Flow"] /
        data["Total Revenue"]
    )

    data["Revenue Growth"] = (
        data["Total Revenue"].pct_change()
    )

    data["Net Income Growth"] = (
        data["Net Income"].pct_change()
    )

    data["Debt Growth"] = (
        data["Total Debt"].pct_change()
    )

    data["Cash Growth"] = (
        data["Cash And Cash Equivalents"].pct_change()
    )

    # Replace infinite values
    data.replace(
        [float("inf"), float("-inf")],
        pd.NA,
        inplace=True
    )

    # Add company identifier
    data["Ticker"] = ticker

    # Reset index so Date becomes a column
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    data.reset_index(inplace=True)

    return data


def main():

    os.makedirs("data/processed", exist_ok=True)

    all_companies = []

    for ticker in TICKERS:

        try:
            company_data = collect_company_data(ticker)

            if company_data is not None:
                all_companies.append(company_data)

                print(
                    f"Successfully collected: {ticker}"
                )

        except Exception as e:
            print(
                f"Error collecting {ticker}: {e}"
            )

        # Small delay between requests
        time.sleep(2)

    # ---------------------------------------------------
    # Combine all companies
    # ---------------------------------------------------

    if not all_companies:
        print("\nNo company data was collected.")
        return

    combined_dataset = pd.concat(
        all_companies,
        ignore_index=True
    )

    # Sort by company and date
    combined_dataset.sort_values(
        ["Ticker", "Date"],
        inplace=True
    )

    # Save dataset
    output_file = (
        "data/processed/"
        "multi_company_financial_dataset.csv"
    )

    combined_dataset.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("MULTI-COMPANY DATASET CREATED")
    print("=" * 60)

    print(f"\nFile: {output_file}")

    print("\nDataset Shape:")
    print(combined_dataset.shape)

    print("\nCompanies:")
    print(
        combined_dataset["Ticker"]
        .unique()
        .tolist()
    )

    print("\nRows per company:")
    print(
        combined_dataset["Ticker"]
        .value_counts()
    )


if __name__ == "__main__":
    main()