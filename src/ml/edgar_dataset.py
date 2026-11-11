"""Modelling dataset built from SEC EDGAR quarterly history.

Input:  data/raw/sec_edgar/quarterly/<TICKER>_quarterly.csv
Output: data/processed/sec_edgar_revenue_decline_features.csv

Labels (only from quarters marked Point_In_Time, 70-110 days apart):
- Next_Period_Revenue_Decline      next quarter's revenue < this quarter's
- Next_Period_Revenue_Decline_YoY  next quarter's revenue < the same
                                   quarter one year earlier (seasonality-free)

Timing columns used by the evaluation:
- Available_Date        this quarter's filing date (when the inputs were public)
- Label_Available_Date  next quarter's filing date (when the label was known)
- Test_Group            nearest calendar quarter end, so companies with
                        different fiscal calendars are evaluated together

Features are the same five ratios as the Yahoo dataset
(add_ratio_features). Note: Total Debt here excludes leases, so the debt
ratios are lower than Yahoo-based ones; this dataset is evaluated on its
own and never mixed with Yahoo rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.prepare_revenue_decline_features import add_ratio_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sec_edgar" / "quarterly"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sec_edgar_revenue_decline_features.csv"
)
QUARTER_GAP = (70, 110)
YEAR_GAP = (350, 380)


def nearest_quarter_end(date):
    date = pd.Timestamp(date)
    candidates = [
        pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
        for y in (date.year - 1, date.year, date.year + 1)
        for m in (3, 6, 9, 12)
    ]
    return min(candidates, key=lambda c: abs((c - date).days))


def label_company(table):
    t = table.copy()
    t["Date"] = pd.to_datetime(t["Date"])
    t["Filed"] = pd.to_datetime(t["Filed"])
    t["Point_In_Time"] = t["Point_In_Time"].astype(str).str.lower() == "true"
    t = t.sort_values("Date").reset_index(drop=True)

    nxt = t.shift(-1)
    gap = (nxt["Date"] - t["Date"]).dt.days
    valid = (
        gap.between(*QUARTER_GAP)
        & t["Point_In_Time"] & nxt["Point_In_Time"].fillna(False).astype(bool)
        & (t["Total Revenue"] > 0) & nxt["Total Revenue"].notna()
    )
    t["Next_Date"] = nxt["Date"]
    t["Next_Revenue"] = nxt["Total Revenue"]
    t["Label_Available_Date"] = nxt["Filed"]
    t["Available_Date"] = t["Filed"]
    t["Next_Period_Revenue_Decline"] = np.where(
        valid, (nxt["Total Revenue"] < t["Total Revenue"]).astype(float), np.nan)

    point_in_time = t[t["Point_In_Time"]]
    yoy = []
    for i, row in t.iterrows():
        if not valid[i]:
            yoy.append(np.nan)
            continue
        days = (row["Next_Date"] - point_in_time["Date"]).dt.days
        base = point_in_time[days.between(*YEAR_GAP)]
        if base.empty or not base.iloc[0]["Total Revenue"] > 0:
            yoy.append(np.nan)
            continue
        yoy.append(float(row["Next_Revenue"] < base.iloc[0]["Total Revenue"]))
    t["Next_Period_Revenue_Decline_YoY"] = yoy

    return t[valid].copy()


def build_dataset(input_dir=INPUT_DIR, tickers=None):
    files = sorted(Path(input_dir).glob("*_quarterly.csv"))
    parts = []
    for path in files:
        ticker = path.stem.replace("_quarterly", "")
        if tickers and ticker not in tickers:
            continue
        parts.append(label_company(pd.read_csv(path)))
    if not parts:
        raise FileNotFoundError(f"No EDGAR quarterly files in {input_dir}")
    data = add_ratio_features(pd.concat(parts, ignore_index=True))
    data["Test_Group"] = data["Date"].map(nearest_quarter_end)
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    for column in ("Date", "Filed", "Next_Date", "Available_Date",
                   "Label_Available_Date", "Test_Group"):
        data[column] = pd.to_datetime(data[column]).dt.date
    return data


def main(input_dir=INPUT_DIR, output_file=OUTPUT_FILE, overwrite=False):
    output_file = Path(output_file)
    if output_file.exists() and not overwrite:
        print("Output already exists:", output_file)
        return None
    data = build_dataset(input_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)
    return data


if __name__ == "__main__":
    main()
