import pandas as pd
import os


# ---------------------------------------------------
# File path
# ---------------------------------------------------

file_path = "data/processed/multi_company_financial_dataset.csv"


# ---------------------------------------------------
# Check whether file exists
# ---------------------------------------------------

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()


# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

data = pd.read_csv(file_path)


print("\n" + "=" * 60)
print("MULTI-COMPANY DATASET INSPECTION")
print("=" * 60)


# ---------------------------------------------------
# 1. Dataset shape
# ---------------------------------------------------

print("\n1. Dataset Shape:")
print(data.shape)


# ---------------------------------------------------
# 2. Column names
# ---------------------------------------------------

print("\n2. Columns:")
print(data.columns.tolist())


# ---------------------------------------------------
# 3. Data types
# ---------------------------------------------------

print("\n3. Data Types:")
print(data.dtypes)


# ---------------------------------------------------
# 4. Companies
# ---------------------------------------------------

print("\n4. Companies:")
print(data["Ticker"].unique().tolist())


# ---------------------------------------------------
# 5. Number of rows per company
# ---------------------------------------------------

print("\n5. Rows Per Company:")
print(data["Ticker"].value_counts())


# ---------------------------------------------------
# 6. Available years
# ---------------------------------------------------

data["Date"] = pd.to_datetime(data["Date"])

print("\n6. Available Years:")
print(sorted(data["Date"].dt.year.unique().tolist()))


# ---------------------------------------------------
# 7. Rows per year
# ---------------------------------------------------

print("\n7. Rows Per Year:")
print(data["Date"].dt.year.value_counts().sort_index())


# ---------------------------------------------------
# 8. Missing values
# ---------------------------------------------------

print("\n8. Missing Values:")
print(data.isnull().sum())


# ---------------------------------------------------
# 9. Missing value percentage
# ---------------------------------------------------

print("\n9. Missing Value Percentage:")

missing_percentage = (
    data.isnull().mean() * 100
).round(2)

print(missing_percentage)


# ---------------------------------------------------
# 10. Duplicate rows
# ---------------------------------------------------

print("\n10. Duplicate Rows:")
print(data.duplicated().sum())


# ---------------------------------------------------
# 11. Duplicate company-year combinations
# ---------------------------------------------------

duplicate_company_year = data.duplicated(
    subset=["Ticker", "Date"]
).sum()

print("\n11. Duplicate Company-Year Rows:")
print(duplicate_company_year)


# ---------------------------------------------------
# 12. Basic statistics
# ---------------------------------------------------

print("\n12. Basic Statistics:")

print(
    data.describe(
        include="all"
    ).transpose()
)


# ---------------------------------------------------
# 13. First 10 rows
# ---------------------------------------------------

print("\n13. First 10 Rows:")
print(data.head(10))


# ---------------------------------------------------
# 14. Last 10 rows
# ---------------------------------------------------

print("\n14. Last 10 Rows:")
print(data.tail(10))


# ---------------------------------------------------
# 15. Dataset information
# ---------------------------------------------------

print("\n15. Dataset Information:")

print(data.info())


print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)