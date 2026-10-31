
from pathlib import Path
import pandas as pd

folder = Path("data/raw/quarterly")
total_col = "Total Debt"
alternative_col = "Long Term Debt And Capital Lease Obligation"

for file in sorted(folder.glob("*_balance_sheet.csv")):
    if "backup" in file.name or "refreshed_test" in file.name:
        continue

    df = pd.read_csv(file)

    if total_col not in df.columns or alternative_col not in df.columns:
        continue

    both = df[[total_col, alternative_col]].dropna()
    if both.empty:
        print(f"\n{file.name}: no comparable values")
        continue

    difference = both[total_col] - both[alternative_col]
    matches = difference.abs() < 1

    print(f"\n--- {file.name} ---")
    print("Quarters with both values:", len(both))
    print("Values matching within 1 currency unit:", int(matches.sum()))
    print("Values differing:", int((~matches).sum()))

    print(
        df.loc[
            both.index,
            ["Date", total_col, alternative_col]
        ].to_string(index=False)
    )
