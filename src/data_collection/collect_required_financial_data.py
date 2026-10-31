from pathlib import Path
import time

import pandas as pd
import yfinance as yf


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "NVDA", "TSLA", "META", "JPM"
]

# Resolve paths relative to the project root, not the terminal's current folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "raw" / "required_financial_data"


def get_metric(data, metric):
    """Return a financial statement row, or NaN values if it is unavailable."""
    if data is None or data.empty or metric not in data.index:
        return pd.Series(
            float("nan"),
            index=data.columns if data is not None else [],
            dtype="float64"
        )

    values = data.loc[metric]

    # Avoid ambiguous results if the source contains duplicate metric labels.
    if isinstance(values, pd.DataFrame):
        values = values.iloc[0]

    return pd.to_numeric(values, errors="coerce")


def get_total_liabilities(balance_data):
    """
    Prefer reported total liabilities.
    Use current + non-current liabilities only when the values reconcile
    with total assets minus stockholders' equity.
    """
    reported = get_metric(
        balance_data, "Total Liabilities Net Minority Interest"
    )

    current = get_metric(balance_data, "Current Liabilities")
    non_current = get_metric(
        balance_data, "Total Non Current Liabilities Net Minority Interest"
    )
    assets = get_metric(balance_data, "Total Assets")
    equity = get_metric(balance_data, "Stockholders Equity")

    result = reported.copy()

    for date in balance_data.columns:
        if pd.notna(reported.loc[date]):
            continue

        values = [
            current.loc[date],
            non_current.loc[date],
            assets.loc[date],
            equity.loc[date],
        ]

        if any(pd.isna(value) for value in values):
            continue

        calculated = current.loc[date] + non_current.loc[date]
        implied = assets.loc[date] - equity.loc[date]
        tolerance = max(1.0, abs(implied) * 1e-6)

        if abs(calculated - implied) <= tolerance:
            result.loc[date] = calculated

    return result


def collect_company_data(ticker):
    """Collect selected income-statement and balance-sheet fields."""
    print(f"\nCollecting required financial data for {ticker}...")

    stock = yf.Ticker(ticker)
    income_data = stock.income_stmt
    balance_data = stock.balance_sheet

    if income_data.empty:
        raise ValueError("The income statement is empty.")

    if balance_data.empty:
        raise ValueError("The balance sheet is empty.")

    income_selected = pd.DataFrame({
        "Total Revenue": get_metric(income_data, "Total Revenue"),
        "Operating Income": get_metric(income_data, "Operating Income"),
        "EBIT": get_metric(income_data, "EBIT"),
        "Net Income": get_metric(income_data, "Net Income"),
        "Interest Expense": get_metric(income_data, "Interest Expense"),
    })

    balance_selected = pd.DataFrame({
        "Total Assets": get_metric(balance_data, "Total Assets"),
        "Current Assets": get_metric(balance_data, "Current Assets"),
        "Current Liabilities": get_metric(balance_data, "Current Liabilities"),
        "Total Liabilities": get_total_liabilities(balance_data),
        "Retained Earnings": get_metric(balance_data, "Retained Earnings"),
        "Total Debt": get_metric(balance_data, "Total Debt"),
        "Cash And Cash Equivalents": get_metric(
            balance_data, "Cash And Cash Equivalents"
        ),
        "Stockholders Equity": get_metric(
            balance_data, "Stockholders Equity"
        ),
    })

    combined = income_selected.join(balance_selected, how="outer")
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "Date"
    combined = combined.reset_index()
    combined["Ticker"] = ticker
    combined = combined.sort_values("Date").reset_index(drop=True)

    key_fields = [
        "Total Assets",
        "Total Liabilities",
        "Stockholders Equity",
    ]

    missing_counts = combined[key_fields].isna().sum()

    # Warn about missing source metrics without printing a warning for every row.
    all_fields = [
        "Total Revenue", "Operating Income", "EBIT", "Net Income",
        "Interest Expense", "Total Assets", "Current Assets",
        "Current Liabilities", "Total Liabilities", "Retained Earnings",
        "Total Debt", "Cash And Cash Equivalents", "Stockholders Equity",
    ]
    missing_fields = [
        field for field in all_fields
        if field not in combined.columns or combined[field].isna().all()
    ]

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_FOLDER / f"{ticker}_required_financial_data.csv"
    combined.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Dataset shape: {combined.shape}")

    if missing_fields:
        print("Metrics entirely missing from the source:", missing_fields)

    print("Missing values in key balance-sheet fields:")
    print(missing_counts.to_string())

    return combined


def main():
    for ticker in TICKERS:
        try:
            collect_company_data(ticker)
        except Exception as error:
            print(f"ERROR collecting {ticker}: {error}")

        time.sleep(2)

    print("\nRequired financial data collection finished.")


if __name__ == "__main__":
    main()
