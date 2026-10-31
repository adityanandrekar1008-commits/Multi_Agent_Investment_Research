from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data" / "processed"

REQUIRED_COLUMNS = [
    "Total Revenue",
    "Net Income",
    "Operating Income",
    "EBITDA",
    "Gross Profit",
    "Total Debt",
    "Cash And Cash Equivalents",
    "Free Cash Flow",
    "Operating Cash Flow",
]


def safe_ratio(numerator, denominator):
    """Calculate a ratio while treating zero denominators as missing."""
    denominator = denominator.where(denominator != 0)
    return numerator / denominator


def annual_growth(series):
    """
    Calculate growth only across consecutive reporting years.
    Growth is undefined when the previous value is zero or negative.
    """
    previous = series.shift(1)
    previous_year = pd.Series(series.index.year, index=series.index).shift(1)
    consecutive_years = (pd.Series(
        series.index.year, index=series.index
    ) - previous_year) == 1

    valid = (
        consecutive_years
        & previous.gt(0)
        & series.notna()
        & previous.notna()
    )

    result = (series / previous) - 1
    return result.where(valid)


def create_financial_features(ticker):
    ticker = ticker.strip().upper()

    if not ticker:
        print("Error: Enter a valid ticker.")
        return

    input_file = DATA_FOLDER / f"{ticker}_combined_financial_data.csv"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    # The first CSV column contains reporting dates.
    data = pd.read_csv(input_file, index_col=0)

    dates = pd.to_datetime(data.index, errors="coerce")
    valid_dates = dates.notna()

    data = data.loc[valid_dates].copy()
    data.index = dates[valid_dates]
    data.index.name = "Date"

    # Convert financial values to numeric.
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.sort_index()
    data = data.dropna(how="all")

    if data.empty:
        print("Error: No usable financial data remains after date cleaning.")
        return

    if data.index.has_duplicates:
        print("Error: Duplicate reporting dates found. Resolve them first.")
        return

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        print("Missing required columns:", missing)
        return

    # Profitability and cash-flow ratios.
    data["Profit Margin"] = safe_ratio(
        data["Net Income"], data["Total Revenue"]
    )
    data["Operating Margin"] = safe_ratio(
        data["Operating Income"], data["Total Revenue"]
    )
    data["EBITDA Margin"] = safe_ratio(
        data["EBITDA"], data["Total Revenue"]
    )
    data["Gross Profit Margin"] = safe_ratio(
        data["Gross Profit"], data["Total Revenue"]
    )
    data["Debt to Cash"] = safe_ratio(
        data["Total Debt"], data["Cash And Cash Equivalents"]
    )
    data["Free Cash Flow Margin"] = safe_ratio(
        data["Free Cash Flow"], data["Total Revenue"]
    )
    data["Operating Cash Flow Margin"] = safe_ratio(
        data["Operating Cash Flow"], data["Total Revenue"]
    )

    # Growth is recorded only for consecutive years with a positive
    # previous value. Net income growth is additionally undefined if
    # either year's net income is zero or negative.
    data["Revenue Growth"] = annual_growth(data["Total Revenue"])

    net_income_growth = annual_growth(data["Net Income"])
    data["Net Income Growth"] = net_income_growth.where(
        data["Net Income"] > 0
    )

    data["Debt Growth"] = annual_growth(data["Total Debt"])
    data["Cash Growth"] = annual_growth(
        data["Cash And Cash Equivalents"]
    )

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    output_file = DATA_FOLDER / f"{ticker}_financial_features.csv"
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index_label="Date")

    print("\nFeature engineering completed.")
    print("Output:", output_file)
    print("Dataset shape:", data.shape)
    print("Reporting dates:", data.index.strftime("%Y-%m-%d").tolist())
    print("\nMissing values per column:")
    print(data.isna().sum().to_string())
    print("\nFeature preview:")
    print(data.tail().to_string())


if __name__ == "__main__":
    ticker = input("Enter company ticker: ").strip().upper()
    create_financial_features(ticker)
