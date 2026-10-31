
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "quarterly"

TICKERS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "RELIANCE.NS",
    "TSLA",
]

STATEMENTS = {
    "income_statement": "quarterly_income_stmt",
    "balance_sheet": "quarterly_balance_sheet",
    "cash_flow": "quarterly_cashflow",
}


def collect_statement(ticker, statement_name, attribute):
    """Collect one quarterly statement without changing missing values."""
    print(f"  Downloading {statement_name}...")

    try:
        statement = getattr(yf.Ticker(ticker), attribute)

        if statement is None or statement.empty:
            print("    No data available.")
            return

        # Convert the statement's date columns into rows.
        result = statement.T.copy()
        result.index.name = "Date"
        result = result.reset_index()
        result["Ticker"] = ticker

        # Put identifying columns first.
        columns = ["Date", "Ticker"] + [
            column
            for column in result.columns
            if column not in ["Date", "Ticker"]
        ]
        result = result[columns]

        output_file = (
            OUTPUT_DIR / f"{ticker}_{statement_name}.csv"
        )

        # Each company/statement gets its own file.
        result.to_csv(output_file, index=False)

        print(f"    Saved {len(result)} periods to {output_file.name}")

    except Exception as exc:
        print(f"    Could not collect data: {exc}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("QUARTERLY FINANCIAL DATA COLLECTION")
    print(f"Output folder: {OUTPUT_DIR}")

    for ticker in TICKERS:
        print(f"\nCompany: {ticker}")

        for statement_name, attribute in STATEMENTS.items():
            collect_statement(ticker, statement_name, attribute)

    print("\nCollection finished.")
    print("Existing annual datasets were not modified.")


if __name__ == "__main__":
    main()
