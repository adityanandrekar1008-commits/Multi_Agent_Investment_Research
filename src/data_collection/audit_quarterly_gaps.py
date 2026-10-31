
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_ml_base_dataset.csv"
)

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

df["Next_Date"] = df.groupby("Ticker")["Date"].shift(-1)
df["Next_Revenue"] = df.groupby("Ticker")["Total Revenue"].shift(-1)
df["Days_To_Next_Report"] = (
    df["Next_Date"] - df["Date"]
).dt.days

df["Next_Revenue_Change_Percent"] = (
    (df["Next_Revenue"] - df["Total Revenue"])
    / df["Total Revenue"]
    * 100
)

print("\n========== REPORTING-DATE GAP AUDIT ==========")
print(
    df[
        [
            "Ticker",
            "Date",
            "Next_Date",
            "Days_To_Next_Report",
            "Total Revenue",
            "Next_Revenue",
            "Next_Revenue_Change_Percent",
        ]
    ].to_string(index=False)
)

print("\n========== GAP SUMMARY ==========")
valid_gaps = df["Days_To_Next_Report"].dropna()

print("Number of report-to-report gaps:", len(valid_gaps))
print("Minimum gap in days:", valid_gaps.min())
print("Maximum gap in days:", valid_gaps.max())

print("\nGaps between 70 and 110 days:")
print(
    int(valid_gaps.between(70, 110).sum())
)

print("\nGaps outside 70–110 days:")
print(
    df.loc[
        df["Days_To_Next_Report"].notna()
        & ~df["Days_To_Next_Report"].between(70, 110),
        ["Ticker", "Date", "Next_Date", "Days_To_Next_Report"],
    ].to_string(index=False)
)

print("\nAudit complete. No files were modified.")
