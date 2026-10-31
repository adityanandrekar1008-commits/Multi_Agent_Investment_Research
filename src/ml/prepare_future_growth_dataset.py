
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_financial_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "future_growth_dataset.csv"
)

# Columns that may be used as model inputs.
# Target-related columns are deliberately excluded.
FEATURE_COLUMNS = [
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "Net Income",
    "EBITDA",
    "Operating Expense",
    "Cost Of Revenue",
    "Research And Development",
    "Interest Expense",
    "Total Debt",
    "Cash And Cash Equivalents",
    "Tangible Book Value",
    "Free Cash Flow",
    "Operating Cash Flow",
    "Capital Expenditure",
    "Profit Margin",
    "Operating Margin",
    "EBITDA Margin",
    "Gross Profit Margin",
    "Debt to Cash",
    "Free Cash Flow Margin",
    "Operating Cash Flow Margin",
    "Debt Growth",
    "Cash Growth",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker"])
    df = df.sort_values(["Ticker", "Date"]).copy()

    # Target: revenue growth in the next available observation.
    df["Target_Next_Revenue_Growth"] = (
        df.groupby("Ticker")["Revenue Growth"].shift(-1)
    )

    # Keep records with a known future target.
    dataset = df.dropna(
        subset=["Target_Next_Revenue_Growth"]
    ).copy()

    dataset["Revenue_Decline_Next_Year"] = (
        dataset["Target_Next_Revenue_Growth"] < 0
    ).astype(int)

    # Select only approved inputs and target columns.
    # Do not include current Revenue Growth or future target
    # as model input features.
    output_columns = (
        ["Date", "Ticker"]
        + FEATURE_COLUMNS
        + ["Revenue_Decline_Next_Year"]
    )

    model_dataset = dataset[output_columns].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_dataset.to_csv(OUTPUT_FILE, index=False)

    print("=" * 55)
    print("ML DATASET PREPARATION")
    print("=" * 55)
    print(f"Records: {len(model_dataset)}")
    print(f"Companies: {model_dataset['Ticker'].nunique()}")
    print(f"Input feature count: {len(FEATURE_COLUMNS)}")

    print("\nTarget class counts:")
    print(
        model_dataset["Revenue_Decline_Next_Year"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values in input features:")
    print(
        model_dataset[FEATURE_COLUMNS]
        .isnull()
        .sum()
        .loc[lambda values: values > 0]
    )

    print("\nSaved dataset:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
