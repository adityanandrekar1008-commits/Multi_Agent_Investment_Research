
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_combined_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_ml_base_dataset.csv"
)


def main():
    if OUTPUT_FILE.exists():
        print("Output already exists:", OUTPUT_FILE)
        print("No file was overwritten.")
        return

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    print("Original rows:", len(df))

    # Exclude the banking company for this initial experiment.
    df = df[df["Ticker"] != "JPM"].copy()

    # Revenue and net income are required for the first target analysis.
    df = df.dropna(
        subset=["Date", "Ticker", "Total Revenue", "Net Income"]
    )

    # Keep only positive revenue values so percentage growth calculations
    # are mathematically meaningful.
    df = df[df["Total Revenue"] > 0].copy()

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Calculate revenue change against the previous available report.
    # This is exploratory only: gaps in reporting dates still need review.
    df["Previous_Revenue"] = (
        df.groupby("Ticker")["Total Revenue"].shift(1)
    )

    df["Revenue_Growth_Percent"] = (
        (
            df["Total Revenue"] - df["Previous_Revenue"]
        )
        / df["Previous_Revenue"]
        * 100
    )

    print("Rows after basic filtering:", len(df))
    print("Companies:", df["Ticker"].nunique())

    print("\nRecords per company:")
    print(df.groupby("Ticker").size().to_string())

    print("\nRevenue growth observations:")
    print(
        df[
            [
                "Date",
                "Ticker",
                "Total Revenue",
                "Previous_Revenue",
                "Revenue_Growth_Percent",
            ]
        ].to_string(index=False)
    )

    print("\nMissing values in calculated revenue growth:")
    print(df["Revenue_Growth_Percent"].isna().sum())

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCreated:", OUTPUT_FILE)
    print("Original source dataset was not modified.")
    print(
        "\nNote: growth compares consecutive available reports, "
        "not necessarily consecutive fiscal quarters."
    )


if __name__ == "__main__":
    main()
