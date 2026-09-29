import pandas as pd
import os


# ---------------------------------------------------
# File paths
# ---------------------------------------------------

cleaned_file = "data/processed/clean_financial_dataset.csv"

raw_folder = "data/raw"


# ---------------------------------------------------
# Check cleaned dataset
# ---------------------------------------------------

if not os.path.exists(cleaned_file):
    print(f"Error: {cleaned_file} not found.")
    exit()


# ---------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------

data = pd.read_csv(cleaned_file)

data["Date"] = pd.to_datetime(data["Date"])


print("\n" + "=" * 60)
print("ALTMAN Z-SCORE DATA AVAILABILITY CHECK")
print("=" * 60)


# ---------------------------------------------------
# Altman Z-score required components
# ---------------------------------------------------

required_components = [
    "Working Capital",
    "Total Assets",
    "Retained Earnings",
    "EBIT",
    "Market Value Of Equity",
    "Total Liabilities",
    "Sales"
]


print("\nRequired Altman Z-Score Components:")

for component in required_components:
    print("-", component)


# ---------------------------------------------------
# Check raw balance sheets for required fields
# ---------------------------------------------------

print("\n" + "=" * 60)
print("CHECKING RAW BALANCE SHEET DATA")
print("=" * 60)


tickers = sorted(data["Ticker"].unique())


for ticker in tickers:

    file_path = (
        f"{raw_folder}/{ticker}_balance_sheet.csv"
    )

    print(f"\n{ticker}:")

    if not os.path.exists(file_path):
        print("  Balance sheet file not found.")
        continue

    balance_sheet = pd.read_csv(
        file_path,
        index_col=0
    )

    available_metrics = balance_sheet.index.tolist()

    print("\n  Available required metrics:")

    for metric in [
        "Total Assets",
        "Retained Earnings",
        "Total Liabilities",
        "Current Assets",
        "Current Liabilities"
    ]:

        if metric in available_metrics:
            print(f"  [YES] {metric}")
        else:
            print(f"  [NO]  {metric}")


# ---------------------------------------------------
# Check income statement metrics
# ---------------------------------------------------

print("\n" + "=" * 60)
print("CHECKING RAW INCOME STATEMENT DATA")
print("=" * 60)


for ticker in tickers:

    file_path = (
        f"{raw_folder}/{ticker}_financial_data.csv"
    )

    print(f"\n{ticker}:")

    if not os.path.exists(file_path):
        print("  Income statement file not found.")
        continue

    income_statement = pd.read_csv(
        file_path,
        index_col=0
    )

    available_metrics = income_statement.index.tolist()

    print("\n  Available required metrics:")

    for metric in [
        "Total Revenue",
        "Operating Income",
        "EBIT",
        "Net Income"
    ]:

        if metric in available_metrics:
            print(f"  [YES] {metric}")
        else:
            print(f"  [NO]  {metric}")


# ---------------------------------------------------
# Final message
# ---------------------------------------------------

print("\n" + "=" * 60)
print("ALTMAN DATA AVAILABILITY CHECK COMPLETED")
print("=" * 60)

print(
    "\nDo not create the Z-score or ML target yet."
)

print(
    "We will use the output to determine which "
    "financial-distress methodology is appropriate."
)