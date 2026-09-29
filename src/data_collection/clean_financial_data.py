import pandas as pd
import os


# ---------------------------------------------------
# File paths
# ---------------------------------------------------

input_file = "data/processed/multi_company_financial_dataset.csv"

output_file = "data/processed/clean_financial_dataset.csv"


# ---------------------------------------------------
# Check input file
# ---------------------------------------------------

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit()


# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

data = pd.read_csv(input_file)

print("\n" + "=" * 60)
print("FINANCIAL DATA CLEANING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(data.shape)


# ---------------------------------------------------
# Convert Date column
# ---------------------------------------------------

data["Date"] = pd.to_datetime(data["Date"])


# ---------------------------------------------------
# Keep annual data from 2022 to 2025
# ---------------------------------------------------

data = data[
    data["Date"].dt.year.between(2022, 2025)
].copy()


print("\nAfter keeping 2022-2025:")
print(data.shape)


# ---------------------------------------------------
# Core financial columns
# ---------------------------------------------------

core_columns = [
    "Total Revenue",
    "Net Income",
    "Total Debt",
    "Cash And Cash Equivalents",
    "Free Cash Flow",
    "Operating Cash Flow"
]


# ---------------------------------------------------
# Remove rows where core financial data is missing
# ---------------------------------------------------

before_removal = len(data)

data = data.dropna(
    subset=core_columns
).copy()

after_removal = len(data)


print("\nRows removed because of missing core financial data:")
print(before_removal - after_removal)


# ---------------------------------------------------
# Sort dataset
# ---------------------------------------------------

data.sort_values(
    ["Ticker", "Date"],
    inplace=True
)


# ---------------------------------------------------
# Reset index
# ---------------------------------------------------

data.reset_index(
    drop=True,
    inplace=True
)


# ---------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

data.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------
# Final information
# ---------------------------------------------------

print("\nCleaned Dataset Shape:")
print(data.shape)


print("\nCompanies:")
print(
    data["Ticker"]
    .unique()
    .tolist()
)


print("\nRows Per Company:")
print(
    data["Ticker"]
    .value_counts()
    .sort_index()
)


print("\nYears:")
print(
    sorted(
        data["Date"]
        .dt.year
        .unique()
        .tolist()
    )
)


print("\nRemaining Missing Values:")

print(
    data.isnull().sum()
)


print("\nCleaned dataset saved successfully!")

print("File:")
print(output_file)


print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)