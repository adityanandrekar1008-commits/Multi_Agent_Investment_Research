from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "multi_company_financial_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "modeling_financial_dataset.csv"

def annual_growth(series, dates):
    previous = series.shift(1)
    previous_dates = dates.shift(1)
    consecutive = (dates.dt.year - previous_dates.dt.year) == 1
    valid = consecutive & previous.gt(0) & series.notna() & previous.notna()
    return ((series / previous) - 1).where(valid)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE)
    print("Original dataset shape:", data.shape)

    required = ["Date", "Ticker"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Ticker"] = data["Ticker"].astype("string").str.strip()
    data = data.dropna(subset=["Date", "Ticker"]).copy()
    data = data[data["Ticker"] != ""].copy()

    financial_columns = [col for col in data.columns if col not in ["Date", "Ticker"]]
    data[financial_columns] = data[financial_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    before = len(data)
    data = data.dropna(subset=financial_columns, how="all").copy()
    print("Completely empty financial rows removed:", before - len(data))

    data = data.sort_values(["Ticker", "Date"]).copy()
    duplicates = data.duplicated(["Ticker", "Date"], keep="last")
    print("Duplicate company-date rows removed:", int(duplicates.sum()))
    data = data.drop_duplicates(["Ticker", "Date"], keep="last").copy()

    def ratio(numerator, denominator):
        if numerator not in data.columns or denominator not in data.columns:
            return pd.Series(np.nan, index=data.index)
        denominator_values = data[denominator].where(data[denominator] != 0)
        return data[numerator] / denominator_values

    ratio_features = {
        "Profit Margin": ("Net Income", "Total Revenue"),
        "Operating Margin": ("Operating Income", "Total Revenue"),
        "EBITDA Margin": ("EBITDA", "Total Revenue"),
        "Gross Profit Margin": ("Gross Profit", "Total Revenue"),
        "Debt to Cash": ("Total Debt", "Cash And Cash Equivalents"),
        "Free Cash Flow Margin": ("Free Cash Flow", "Total Revenue"),
        "Operating Cash Flow Margin": ("Operating Cash Flow", "Total Revenue"),
    }

    for feature, (numerator, denominator) in ratio_features.items():
        data[feature] = ratio(numerator, denominator)

    growth_features = {
        "Revenue Growth": "Total Revenue",
        "Net Income Growth": "Net Income",
        "Debt Growth": "Total Debt",
        "Cash Growth": "Cash And Cash Equivalents",
    }

    for feature, source_column in growth_features.items():
        if source_column in data.columns:
            data[feature] = data.groupby("Ticker", group_keys=False).apply(
                lambda group: annual_growth(group[source_column], group["Date"]),
                include_groups=False
            ).sort_index()
        else:
            data[feature] = np.nan

    if "Net Income" in data.columns:
        data["Net Income Growth"] = data["Net Income Growth"].where(
            data["Net Income"] > 0
        )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print("\nClean modeling dataset shape:", data.shape)
    print("\nCompanies and row counts:")
    print(data.groupby("Ticker").size().to_string())
    print("\nYears:", sorted(data["Date"].dt.year.unique().tolist()))
    print("\nMissing values in key columns:")
    key_columns = [
        col for col in [
            "Total Revenue", "Net Income", "Total Debt",
            "Cash And Cash Equivalents", "Free Cash Flow",
            "Operating Cash Flow", "Profit Margin", "Revenue Growth"
        ] if col in data.columns
    ]
    print(data[key_columns].isna().sum().to_string())
    print("\nSaved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
