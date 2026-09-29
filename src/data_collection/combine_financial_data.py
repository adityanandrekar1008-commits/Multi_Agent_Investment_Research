import pandas as pd
import os


def load_statement(file_path):
    return pd.read_csv(file_path, index_col=0)


def get_metric(data, metric):
    if metric in data.index:
        return data.loc[metric]
    else:
        return pd.Series(index=data.columns, dtype="float64")


if __name__ == "__main__":

    ticker = input("Enter company ticker: ").upper()

    income_file = f"data/raw/{ticker}_financial_data.csv"
    balance_file = f"data/raw/{ticker}_balance_sheet.csv"
    cashflow_file = f"data/raw/{ticker}_cash_flow.csv"

    # Check whether files exist
    if not os.path.exists(income_file):
        print(f"Error: {income_file} not found.")
        exit()

    if not os.path.exists(balance_file):
        print(f"Error: {balance_file} not found.")
        exit()

    if not os.path.exists(cashflow_file):
        print(f"Error: {cashflow_file} not found.")
        exit()

    # Load financial statements
    income_data = load_statement(income_file)
    balance_data = load_statement(balance_file)
    cashflow_data = load_statement(cashflow_file)

    # Select important financial metrics
    selected_data = {
        "Total Revenue": get_metric(income_data, "Total Revenue"),
        "Gross Profit": get_metric(income_data, "Gross Profit"),
        "Operating Income": get_metric(income_data, "Operating Income"),
        "Net Income": get_metric(income_data, "Net Income"),
        "EBITDA": get_metric(income_data, "EBITDA"),
        "Operating Expense": get_metric(income_data, "Operating Expense"),
        "Cost Of Revenue": get_metric(income_data, "Cost Of Revenue"),
        "Research And Development": get_metric(
            income_data, "Research And Development"
        ),
        "Interest Expense": get_metric(income_data, "Interest Expense"),

        "Total Debt": get_metric(balance_data, "Total Debt"),
        "Cash And Cash Equivalents": get_metric(
            balance_data, "Cash And Cash Equivalents"
        ),
        "Tangible Book Value": get_metric(
            balance_data, "Tangible Book Value"
        ),

        "Free Cash Flow": get_metric(cashflow_data, "Free Cash Flow"),
        "Operating Cash Flow": get_metric(
            cashflow_data, "Operating Cash Flow"
        ),
        "Capital Expenditure": get_metric(
            cashflow_data, "Capital Expenditure"
        ),
    }

    # Create combined dataset
    combined_data = pd.DataFrame(selected_data)

    # Convert date columns into a normal Date column
    combined_data.index = pd.to_datetime(combined_data.index)
    combined_data.index.name = "Date"

    # Sort by date
    combined_data = combined_data.sort_index()

    # Save combined dataset
    os.makedirs("data/processed", exist_ok=True)

    output_file = f"data/processed/{ticker}_combined_financial_data.csv"

    combined_data.to_csv(output_file)

    print("\nCombined financial dataset created successfully!")
    print("File:", output_file)

    print("\nDataset Shape:")
    print(combined_data.shape)

    print("\nCombined Dataset:")
    print(combined_data)