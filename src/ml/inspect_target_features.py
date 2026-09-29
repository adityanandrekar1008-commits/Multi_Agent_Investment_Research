import pandas as pd
import os


# ---------------------------------------------------
# File path
# ---------------------------------------------------

input_file = "data/processed/clean_financial_dataset.csv"


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

data["Date"] = pd.to_datetime(data["Date"])


print("\n" + "=" * 60)
print("ML TARGET FEATURE INSPECTION")
print("=" * 60)


# ---------------------------------------------------
# Features that may be relevant to financial distress
# ---------------------------------------------------

target_features = [
    "Profit Margin",
    "Operating Margin",
    "EBITDA Margin",
    "Gross Profit Margin",
    "Debt to Cash",
    "Free Cash Flow Margin",
    "Operating Cash Flow Margin",
    "Revenue Growth",
    "Net Income Growth",
    "Debt Growth",
    "Cash Growth"
]


# ---------------------------------------------------
# Display statistics
# ---------------------------------------------------

print("\n1. Target-Related Feature Statistics:")
print(
    data[target_features]
    .describe()
    .transpose()
)


# ---------------------------------------------------
# Negative Net Income
# ---------------------------------------------------

data["Negative Net Income"] = (
    data["Net Income"] < 0
).astype(int)

print("\n2. Negative Net Income Count:")
print(
    data["Negative Net Income"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------
# Negative Free Cash Flow
# ---------------------------------------------------

data["Negative Free Cash Flow"] = (
    data["Free Cash Flow"] < 0
).astype(int)

print("\n3. Negative Free Cash Flow Count:")
print(
    data["Negative Free Cash Flow"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------
# Negative Operating Cash Flow
# ---------------------------------------------------

data["Negative Operating Cash Flow"] = (
    data["Operating Cash Flow"] < 0
).astype(int)

print("\n4. Negative Operating Cash Flow Count:")
print(
    data["Negative Operating Cash Flow"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------
# Negative Revenue Growth
# ---------------------------------------------------

data["Negative Revenue Growth"] = (
    data["Revenue Growth"] < 0
).astype(int)

print("\n5. Negative Revenue Growth Count:")
print(
    data["Negative Revenue Growth"]
    .value_counts(dropna=False)
    .sort_index()
)


# ---------------------------------------------------
# Negative Net Income Growth
# ---------------------------------------------------

data["Negative Net Income Growth"] = (
    data["Net Income Growth"] < 0
).astype(int)

print("\n6. Negative Net Income Growth Count:")
print(
    data["Negative Net Income Growth"]
    .value_counts(dropna=False)
    .sort_index()
)


# ---------------------------------------------------
# High Debt-to-Cash observations
# ---------------------------------------------------

print("\n7. Debt-to-Cash Statistics:")

print(
    data["Debt to Cash"]
    .describe()
)


# ---------------------------------------------------
# Company-wise financial health features
# ---------------------------------------------------

print("\n8. Company-Wise Summary:")

company_summary = (
    data.groupby("Ticker")[target_features]
    .mean()
    .round(3)
)

print(company_summary)


# ---------------------------------------------------
# Year-wise summary
# ---------------------------------------------------

print("\n9. Year-Wise Summary:")

year_summary = (
    data.groupby(
        data["Date"].dt.year
    )[target_features]
    .mean()
    .round(3)
)

print(year_summary)


print("\n" + "=" * 60)
print("TARGET FEATURE INSPECTION COMPLETED")
print("=" * 60)