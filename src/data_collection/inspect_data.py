import pandas as pd


file_path = "data/raw/TSLA_financial_data.csv"

data = pd.read_csv(file_path, index_col=0)

print("\nDataset Shape:")
print(data.shape)

print("\nFinancial Metrics:")
print(data.index.tolist())

print("\nAvailable Years:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nFirst 5 Rows:")
print(data.head())