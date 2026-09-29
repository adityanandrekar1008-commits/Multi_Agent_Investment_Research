import pandas as pd
import os


def create_financial_features(ticker):

    input_file = f"data/processed/{ticker}_combined_financial_data.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load combined financial data
    data = pd.read_csv(input_file, index_col=0)

    # Convert columns to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # ---------------------------------------------------
    # 1. Profit Margin
    # ---------------------------------------------------
    data["Profit Margin"] = (
        data["Net Income"] / data["Total Revenue"]
    )

    # ---------------------------------------------------
    # 2. Operating Margin
    # ---------------------------------------------------
    data["Operating Margin"] = (
        data["Operating Income"] / data["Total Revenue"]
    )

    # ---------------------------------------------------
    # 3. EBITDA Margin
    # ---------------------------------------------------
    data["EBITDA Margin"] = (
        data["EBITDA"] / data["Total Revenue"]
    )

    # ---------------------------------------------------
    # 4. Gross Profit Margin
    # ---------------------------------------------------
    data["Gross Profit Margin"] = (
        data["Gross Profit"] / data["Total Revenue"]
    )

    # ---------------------------------------------------
    # 5. Debt-to-Cash Ratio
    # ---------------------------------------------------
    data["Debt to Cash"] = (
        data["Total Debt"] / data["Cash And Cash Equivalents"]
    )

    # ---------------------------------------------------
    # 6. Free Cash Flow Margin
    # ---------------------------------------------------
    data["Free Cash Flow Margin"] = (
        data["Free Cash Flow"] / data["Total Revenue"]
    )

    # ---------------------------------------------------
    # 7. Operating Cash Flow Margin
    # ---------------------------------------------------
    data["Operating Cash Flow Margin"] = (
        data["Operating Cash Flow"] / data["Total Revenue"]
    )

    # ---------------------------------------------------
    # 8. Revenue Growth
    # ---------------------------------------------------
    data["Revenue Growth"] = data["Total Revenue"].pct_change()

    # ---------------------------------------------------
    # 9. Net Income Growth
    # ---------------------------------------------------
    data["Net Income Growth"] = data["Net Income"].pct_change()

    # ---------------------------------------------------
    # 10. Debt Growth
    # ---------------------------------------------------
    data["Debt Growth"] = data["Total Debt"].pct_change()

    # ---------------------------------------------------
    # 11. Cash Growth
    # ---------------------------------------------------
    data["Cash Growth"] = (
        data["Cash And Cash Equivalents"].pct_change()
    )

    # ---------------------------------------------------
    # Replace infinite values with missing values
    # ---------------------------------------------------
    data.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    # ---------------------------------------------------
    # Save engineered dataset
    # ---------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)

    output_file = (
        f"data/processed/{ticker}_financial_features.csv"
    )

    data.to_csv(output_file)

    print("\nFinancial feature engineering completed!")
    print("File:", output_file)

    print("\nDataset Shape:")
    print(data.shape)

    print("\nFinancial Features:")
    print(data.columns.tolist())

    print("\nFeature Dataset:")
    print(data)


if __name__ == "__main__":

    ticker = input("Enter company ticker: ").upper()

    create_financial_features(ticker)