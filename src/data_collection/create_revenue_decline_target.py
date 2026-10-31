
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_ml_base_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_revenue_decline_target.csv"
)


def main():
    if OUTPUT_FILE.exists():
        print("Output already exists:", OUTPUT_FILE)
        print("No file was overwritten.")
        return

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Get the next reported revenue and date within each company.
    grouped = df.groupby("Ticker", sort=False)

    df["Next_Date"] = grouped["Date"].shift(-1)
    df["Next_Revenue"] = grouped["Total Revenue"].shift(-1)

    df["Days_To_Next_Report"] = (
        df["Next_Date"] - df["Date"]
    ).dt.days

    # Keep only valid, approximately quarterly transitions.
    valid = (
        df["Date"].notna()
        & df["Next_Date"].notna()
        & df["Total Revenue"].notna()
        & df["Next_Revenue"].notna()
        & df["Total Revenue"].gt(0)
        & df["Days_To_Next_Report"].between(70, 110)
    )

    result = df.loc[valid].copy()

    # 1 means next reported revenue is lower; otherwise 0.
    result["Next_Period_Revenue_Decline"] = (
        result["Next_Revenue"] < result["Total Revenue"]
    ).astype(int)

    result["Next_Revenue_Change_Percent"] = (
        (
            result["Next_Revenue"] - result["Total Revenue"]
        )
        / result["Total Revenue"]
        * 100
    )

    result = result.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== TARGET DATASET CREATED ==========")
    print("Output:", OUTPUT_FILE)
    print("Labeled transitions:", len(result))
    print("Companies:", result["Ticker"].nunique())

    print("\nTarget distribution:")
    print(
        result["Next_Period_Revenue_Decline"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No decline (0)", 1: "Decline (1)"})
        .to_string()
    )

    print("\nDecline percentage:")
    print(
        round(
            result["Next_Period_Revenue_Decline"].mean() * 100,
            2,
        ),
        "%",
    )

    print("\nLabeled transitions:")
    print(
        result[
            [
                "Ticker",
                "Date",
                "Next_Date",
                "Days_To_Next_Report",
                "Total Revenue",
                "Next_Revenue",
                "Next_Revenue_Change_Percent",
                "Next_Period_Revenue_Decline",
            ]
        ].to_string(index=False)
    )

    print("\nImportant:")
    print("- Next_Revenue is an outcome/audit column, not an input feature.")
    print("- Next_Date and Days_To_Next_Report are also not input features.")
    print("- The dataset is for experimentation, not reliable risk prediction.")
    print("- The original dataset was not modified.")


if __name__ == "__main__":
    main()
