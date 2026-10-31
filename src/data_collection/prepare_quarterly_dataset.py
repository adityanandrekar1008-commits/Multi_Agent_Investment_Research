
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "quarterly"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "quarterly_combined_dataset.csv"

# Keep selected fields that are useful for an initial financial analysis.
SELECTED_FIELDS = {
    "income_statement": [
        "Total Revenue",
        "Operating Revenue",
        "Net Income",
        "Operating Income",
        "EBITDA",
        "EBIT",
        "Gross Profit",
        "Cost Of Revenue",
        "Operating Expense",
        "Research And Development",
        "Interest Expense",
    ],
    "balance_sheet": [
        "Total Assets",
        "Total Debt",
        "Stockholders Equity",
        "Total Liabilities Net Minority Interest",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Current Assets",
        "Current Liabilities",
        "Working Capital",
    ],
    "cash_flow": [
        "Operating Cash Flow",
        "Free Cash Flow",
        "Capital Expenditure",
        "End Cash Position",
        "Changes In Cash",
    ],
}


def load_statement(ticker, statement_type):
    """Load one statement and keep only selected available fields."""
    file_path = INPUT_DIR / f"{ticker}_{statement_type}.csv"

    if not file_path.exists():
        print(f"  Missing file: {file_path.name}")
        return None

    data = pd.read_csv(file_path)

    if data.empty or "Date" not in data.columns:
        print(f"  Empty or invalid file: {file_path.name}")
        return None

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Ticker"])

    selected = [
        column
        for column in SELECTED_FIELDS[statement_type]
        if column in data.columns
    ]

    # Do not include empty fields that are missing in every row.
    selected = [
        column for column in selected
        if data[column].notna().any()
    ]

    result = data[["Date", "Ticker"] + selected].copy()

    # Prevent accidental many-to-many merges.
    result = result.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last",
    )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(f"Output already exists: {OUTPUT_FILE}")
        print("No files were changed. Rename or move the output file")
        print("if you intentionally want to create it again.")
        return

    income_files = sorted(INPUT_DIR.glob("*_income_statement.csv"))

    if not income_files:
        print("No quarterly income statement files were found.")
        return

    combined_parts = []

    for income_file in income_files:
        ticker = income_file.name.replace(
            "_income_statement.csv", ""
        )

        print(f"\nProcessing {ticker}...")

        income = load_statement(ticker, "income_statement")

        if income is None:
            continue

        # Income statement is the base: retain its reporting dates.
        combined = income

        for statement_type in ["balance_sheet", "cash_flow"]:
            extra = load_statement(ticker, statement_type)

            if extra is None:
                continue

            combined = combined.merge(
                extra,
                on=["Date", "Ticker"],
                how="left",
                validate="one_to_one",
            )

        combined_parts.append(combined)
        print("  Income statement periods:", len(income))
        print("  Combined rows:", len(combined))

    if not combined_parts:
        print("No usable income statement data was found.")
        return

    final_data = pd.concat(combined_parts, ignore_index=True)
    final_data = final_data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    if final_data.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Duplicate company/date records detected.")

    # Save only to a new output file; source files remain unchanged.
    final_data.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 55)
    print("QUARTERLY DATASET CREATED")
    print("Output:", OUTPUT_FILE)
    print("Rows:", len(final_data))
    print("Columns:", len(final_data.columns))
    print("Companies:", final_data["Ticker"].nunique())
    print("Date range:", final_data["Date"].min(),
          "to", final_data["Date"].max())
    print("Duplicate company/date records:",
          final_data.duplicated(["Ticker", "Date"]).sum())
    print("\nMissing values by selected field:")
    print(final_data.isna().sum().to_string())
    print("\nFirst 10 records:")
    print(final_data.head(10).to_string(index=False))
    print("\nOriginal annual and quarterly source files were not modified.")


if __name__ == "__main__":
    main()
