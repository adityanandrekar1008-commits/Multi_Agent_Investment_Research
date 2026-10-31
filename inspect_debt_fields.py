
from pathlib import Path
import pandas as pd

folder = Path("data/raw/quarterly")

fields = [
    "Total Debt",
    "Long Term Debt And Capital Lease Obligation",
    "Long Term Debt",
    "Current Debt And Capital Lease Obligation",
]

for file in sorted(folder.glob("*_balance_sheet.csv")):
    if "backup" in file.name or "refreshed_test" in file.name:
        continue

    data = pd.read_csv(file)

    print(f"\n--- {file.name} ---")

    for field in fields:
        if field in data.columns:
            available = data[field].notna().sum()
            total = len(data)
            print(f"{field}: {available}/{total} values available")
