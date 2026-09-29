import pandas as pd
import os


# ---------------------------------------------------
# Folder containing required financial data
# ---------------------------------------------------

data_folder = "data/raw/required_financial_data"


# ---------------------------------------------------
# Companies
# ---------------------------------------------------

tickers = [
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
# Required financial components
# ---------------------------------------------------

required_columns = [
    "Total Revenue",
    "EBIT",
    "Total Assets",
    "Current Assets",
    "Current Liabilities",
    "Total Liabilities",
    "Retained Earnings",
    "Stockholders Equity"
]


print("\n" + "=" * 60)
print("FINANCIAL COMPONENT VERIFICATION")
print("=" * 60)


# ---------------------------------------------------
# Store all company data
# ---------------------------------------------------

all_data = []


# ---------------------------------------------------
# Check every company
# ---------------------------------------------------

for ticker in tickers:

    file_path = (
        f"{data_folder}/"
        f"{ticker}_required_financial_data.csv"
    )

    print("\n" + "-" * 60)
    print(f"Company: {ticker}")
    print("-" * 60)

    if not os.path.exists(file_path):

        print("File not found!")
        continue


    data = pd.read_csv(file_path)

    all_data.append(data)


    print("Rows:", len(data))

    print("\nRequired columns:")

    for column in required_columns:

        if column in data.columns:

            non_null = data[column].notna().sum()

            print(
                f"[YES] {column} "
                f"-> {non_null}/{len(data)} values"
            )

        else:

            print(
                f"[NO]  {column}"
            )


# ---------------------------------------------------
# Combine all companies
# ---------------------------------------------------

if not all_data:

    print("\nNo data found.")
    exit()


combined_data = pd.concat(
    all_data,
    ignore_index=True
)


# ---------------------------------------------------
# Convert Date
# ---------------------------------------------------

combined_data["Date"] = pd.to_datetime(
    combined_data["Date"]
)


# ---------------------------------------------------
# Overall availability
# ---------------------------------------------------

print("\n" + "=" * 60)
print("OVERALL DATA AVAILABILITY")
print("=" * 60)


for column in required_columns:

    if column in combined_data.columns:

        total_rows = len(combined_data)

        available = (
            combined_data[column]
            .notna()
            .sum()
        )

        percentage = (
            available / total_rows
        ) * 100

        print(
            f"{column}: "
            f"{available}/{total_rows} "
            f"({percentage:.1f}%)"
        )

    else:

        print(
            f"{column}: COLUMN NOT FOUND"
        )


# ---------------------------------------------------
# Missing values
# ---------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)


print(
    combined_data[
        required_columns
    ].isnull().sum()
)


# ---------------------------------------------------
# Years
# ---------------------------------------------------

print("\n" + "=" * 60)
print("AVAILABLE YEARS")
print("=" * 60)

print(
    sorted(
        combined_data["Date"]
        .dt.year
        .unique()
        .tolist()
    )
)


# ---------------------------------------------------
# Rows per company
# ---------------------------------------------------

print("\n" + "=" * 60)
print("ROWS PER COMPANY")
print("=" * 60)

print(
    combined_data["Ticker"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------
# Final dataset shape
# ---------------------------------------------------

print("\n" + "=" * 60)
print("COMBINED DATASET")
print("=" * 60)

print(
    "Shape:",
    combined_data.shape
)


print("\nColumns:")
print(
    combined_data.columns.tolist()
)


print("\n" + "=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)

print(
    "\nDo not create the ML target yet."
)