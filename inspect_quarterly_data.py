
from pathlib import Path
import pandas as pd

folder = Path("data/raw/quarterly")
files = sorted(folder.glob("*.csv"))

print("Total CSV files:", len(files))

for file in files:
    data = pd.read_csv(file)

    print(f"\n{'=' * 60}")
    print("File:", file.name)
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    if "Date" in data.columns:
        dates = pd.to_datetime(data["Date"], errors="coerce")
        print("Date range:", dates.min(), "to", dates.max())

    print("Fields:")
    print(list(data.columns))

print("\nInspection completed. No files were modified.")
