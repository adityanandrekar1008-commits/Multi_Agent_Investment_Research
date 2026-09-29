import pandas as pd


file_path = "data/raw/required_financial_data/AAPL_required_financial_data.csv"

data = pd.read_csv(file_path)


print("\n" + "=" * 60)
print("AAPL FINANCIAL FIELDS")
print("=" * 60)


print("\nAll available financial fields:\n")

for field in data.columns:
    print(field)


print("\n" + "=" * 60)
print("FIELDS RELATED TO LIABILITIES")
print("=" * 60)


for field in data.columns:

    field_lower = field.lower()

    if "liabil" in field_lower:
        print(field)


print("\n" + "=" * 60)
print("FIELDS RELATED TO ASSETS")
print("=" * 60)


for field in data.columns:

    field_lower = field.lower()

    if "asset" in field_lower:
        print(field)


print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)