
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "revenue_decline_features.csv"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    target = "Next_Period_Revenue_Decline"

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("Companies:", df["Ticker"].nunique())
    print("Date range:", df["Date"].min(), "to", df["Date"].max())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df[target].value_counts().sort_index())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes.to_string())

    print("\n========== MISSING VALUES ==========")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() or "No missing values")

    print("\n========== TARGET BY COMPANY ==========")
    print(
        pd.crosstab(df["Ticker"], df[target]).to_string()
    )

    print("\n========== DUPLICATE CHECK ==========")
    print(
        "Duplicate company/date rows:",
        df.duplicated(subset=["Ticker", "Date"]).sum()
    )

    print("\n========== CHECK COMPLETE ==========")
    print("No data files were modified.")


if __name__ == "__main__":
    main()
