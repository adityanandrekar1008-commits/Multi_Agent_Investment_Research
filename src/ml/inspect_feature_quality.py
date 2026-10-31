
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "revenue_decline_features.csv"
)

FEATURES = [
    "Net_Income_Margin",
    "Operating_Margin",
    "Debt_To_Assets",
    "Debt_To_Equity",
    "Current_Ratio",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    print("\n========== FEATURE QUALITY REPORT ==========")
    print("Dataset shape:", df.shape)

    print("\n========== FEATURE STATISTICS ==========")
    print(
        df[FEATURES]
        .describe()
        .round(3)
        .to_string()
    )

    print("\n========== EXTREME VALUES ==========")

    for feature in FEATURES:
        values = df[feature].dropna()

        if values.empty:
            print(f"\n{feature}: no available values")
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        unusual = df[
            (df[feature] < lower)
            | (df[feature] > upper)
        ][["Ticker", "Date", feature]]

        print(f"\n{feature}")
        print(f"  Lower screening bound: {lower:.3f}")
        print(f"  Upper screening bound: {upper:.3f}")
        print(f"  Flagged observations: {len(unusual)}")

        if not unusual.empty:
            print(unusual.to_string(index=False))

    print("\n========== MISSING VALUES BY FEATURE ==========")
    print(df[FEATURES].isna().sum().to_string())

    print("\n========== FEATURE QUALITY CHECK COMPLETE ==========")
    print("No values were changed or removed.")
    print("Screening bounds flag unusual values; they do not prove errors.")


if __name__ == "__main__":
    main()
