
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "quarterly_revenue_decline_target.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "revenue_decline_features.csv"
)

TARGET = "Next_Period_Revenue_Decline"

# Only current-period financial data is used as input.
BASE_FEATURES = [
    "Total Revenue",
    "Net Income",
    "Operating Income",
    "EBITDA",
    "Total Assets",
    "Total Debt",
    "Stockholders Equity",
    "Total Liabilities Net Minority Interest",
    "Cash And Cash Equivalents",
    "Current Assets",
    "Current Liabilities",
    "Operating Cash Flow",
    "Free Cash Flow",
]


def main():
    if OUTPUT_FILE.exists():
        print("Output already exists:", OUTPUT_FILE)
        print("No file was overwritten.")
        return

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    available = [col for col in BASE_FEATURES if col in df.columns]

    features = df[
        ["Date", "Ticker", TARGET] + available
    ].copy()

    # Calculate ratios from the current reporting period only.
    revenue = features["Total Revenue"].where(
        features["Total Revenue"] > 0
    )

    if "Net Income" in features:
        features["Net_Income_Margin"] = (
            features["Net Income"] / revenue
        )

    if "Operating Income" in features:
        features["Operating_Margin"] = (
            features["Operating Income"] / revenue
        )

    if "Total Assets" in features:
        assets = features["Total Assets"].where(
            features["Total Assets"] > 0
        )
        if "Total Debt" in features:
            features["Debt_To_Assets"] = (
                features["Total Debt"] / assets
            )

    if (
        "Total Debt" in features
        and "Stockholders Equity" in features
    ):
        equity = features["Stockholders Equity"].where(
            features["Stockholders Equity"] > 0
        )
        features["Debt_To_Equity"] = (
            features["Total Debt"] / equity
        )

    if (
        "Current Assets" in features
        and "Current Liabilities" in features
    ):
        liabilities = features["Current Liabilities"].where(
            features["Current Liabilities"] > 0
        )
        features["Current_Ratio"] = (
            features["Current Assets"] / liabilities
        )

    features.to_csv(OUTPUT_FILE, index=False)

    print("\n========== FEATURE DATASET CREATED ==========")
    print("Output:", OUTPUT_FILE)
    print("Rows:", len(features))
    print("Companies:", features["Ticker"].nunique())
    print("Target counts:")
    print(features[TARGET].value_counts().sort_index().to_string())

    print("\nFeature missing-value percentages:")
    model_columns = [
        col for col in features.columns
        if col not in ["Date", "Ticker", TARGET]
    ]
    missing = (
        features[model_columns]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    print(missing.to_string())

    print("\nColumns:")
    print(features.columns.tolist())

    print("\nFuture outcome columns were excluded from the feature set.")
    print("Missing values have not been imputed.")
    print("No original files were modified.")


if __name__ == "__main__":
    main()
