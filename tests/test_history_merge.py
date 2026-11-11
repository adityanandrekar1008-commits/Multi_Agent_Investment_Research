"""Merging fresh downloads with stored quarterly history.

Run: python -m unittest discover -s tests -t .
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.rebuild import merge_raw_history


class HistoryMergeTests(unittest.TestCase):
    def test_fields_periods_and_restatements(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_dir, history_dir, out_dir = (
                Path(tmp) / "fresh", Path(tmp) / "history", Path(tmp) / "out"
            )
            fresh_dir.mkdir()
            history_dir.mkdir()

            # History: three full quarters.
            pd.DataFrame({
                "Date": ["2025-12-31", "2025-09-30", "2025-06-30"],
                "Ticker": "X",
                "Total Revenue": [120.0, 110.0, 100.0],
                "Net Income": [12.0, 11.0, 10.0],
            }).to_csv(history_dir / "X_income_statement.csv", index=False)

            # Fresh: a new quarter, a restated value, and a near-empty
            # oldest column (Yahoo's usual pattern).
            pd.DataFrame({
                "Date": ["2026-03-31", "2025-12-31", "2025-09-30"],
                "Ticker": "X",
                "Total Revenue": [130.0, 121.0, None],
                "Net Income": [13.0, 12.0, None],
            }).to_csv(fresh_dir / "X_income_statement.csv", index=False)

            summary = merge_raw_history(fresh_dir, history_dir, out_dir, ["X"])
            merged = pd.read_csv(out_dir / "X_income_statement.csv").set_index("Date")

        self.assertEqual(list(merged.index),
                         ["2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
        self.assertEqual(merged.loc["2025-12-31", "Total Revenue"], 121.0)  # fresh wins
        self.assertEqual(merged.loc["2025-09-30", "Total Revenue"], 110.0)  # history fills
        self.assertEqual(merged.loc["2025-06-30", "Net Income"], 10.0)      # history only
        self.assertEqual(list(merged.columns[:1]), ["Ticker"])
        stats = summary["X_income_statement.csv"]
        self.assertEqual(stats["history_only"], 1)
        self.assertEqual(stats["fields_filled"], 2)


if __name__ == "__main__":
    unittest.main()
