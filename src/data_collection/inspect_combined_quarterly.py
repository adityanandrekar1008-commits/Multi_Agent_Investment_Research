
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_combined_dataset.csv"
)

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Companies:", df["Ticker"].nunique())
print("Date range:", df["Date"].min(), "to", df["Date"].max())

print("\n========== RECORDS PER COMPANY ==========")
print(df.groupby("Ticker").size().to_string())

print("\n========== MISSING VALUE PERCENTAGES ==========")
missing_pct = (
    df.isna().mean().mul(100).round(1)
    .sort_values(ascending=False)
)
print(missing_pct[missing_pct > 0].to_string())

print("\n========== CHECK DUPLICATE RECORDS ==========")
print(
    "Duplicates:",
    df.duplicated(["Ticker", "Date"]).sum()
)

print("\n========== REVENUE AND NET INCOME ==========")
columns = [
    "Date",
    "Ticker",
    "Total Revenue",
    "Net Income",
    "Operating Income",
    "Total Debt",
    "Total Assets",
    "Stockholders Equity",
]

available = [col for col in columns if col in df.columns]

print(
    df[available]
    .sort_values(["Ticker", "Date"])
    .to_string(index=False)
)

print("\n========== FINANCIAL FIELD VALIDATION ==========")
for col in [
    "Total Revenue",
    "Net Income",
    "Total Assets",
    "Total Debt",
    "Stockholders Equity",
]:
    if col in df.columns:
        print(
            f"{col}:",
            "available =", int(df[col].notna().sum()),
            "| negative =", int((df[col] < 0).sum()),
            "| zero =", int((df[col] == 0).sum()),
        )

print("\nInspection complete. No files were modified.")
