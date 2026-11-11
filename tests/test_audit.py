"""Data-quality audit checks on small synthetic datasets.

Run: python -m unittest discover -s tests -t .
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_quality.audit import audit_quarterly


def company(ticker, dates, **overrides):
    rows = []
    for date in dates:
        row = {
            "Date": date, "Ticker": ticker,
            "Total Revenue": 100.0, "Net Income": 10.0, "Operating Income": 15.0,
            "Total Assets": 500.0, "Total Debt": 100.0,
            "Stockholders Equity": 200.0,
            "Total Liabilities Net Minority Interest": 300.0,
            "Current Assets": 150.0, "Current Liabilities": 100.0,
            "Working Capital": 50.0, "Operating Cash Flow": 20.0,
            "Free Cash Flow": 12.0, "Capital Expenditure": -8.0,
        }
        row.update(overrides.get(date, {}))
        rows.append(row)
    return pd.DataFrame(rows)


QUARTERS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31",
            "2026-03-31", "2026-06-30", "2026-09-30", "2026-12-31"]


def checks(result, ticker=None):
    return {f["check"] for f in result["findings"]
            if ticker is None or f["ticker"] == ticker}


class AuditTests(unittest.TestCase):
    def test_clean_company_has_no_warnings(self):
        result = audit_quarterly(company("OK", QUARTERS), as_of="2027-01-15")
        self.assertEqual(result["finding_counts"]["warning"], 0)
        self.assertEqual(result["finding_counts"]["error"], 0)

    def test_gap_stale_and_short_history(self):
        data = company("GAP", ["2025-03-31", "2025-09-30", "2025-12-31"])
        result = audit_quarterly(data, as_of="2026-11-06")
        found = checks(result, "GAP")
        self.assertIn("quarterly_gap", found)
        self.assertIn("stale_data", found)
        self.assertIn("short_history", found)

    def test_duplicates_negative_equity_and_jumps(self):
        data = company("BAD", QUARTERS, **{
            "2025-09-30": {"Stockholders Equity": -50.0},
            "2025-12-31": {"Total Revenue": 300.0},
        })
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        result = audit_quarterly(data, as_of="2027-01-15")
        found = checks(result, "BAD")
        self.assertIn("duplicate_period", found)
        self.assertIn("negative_equity", found)
        self.assertIn("revenue_jump", found)

    def test_incomplete_latest_period(self):
        data = company("NEW", QUARTERS, **{
            "2026-12-31": {"Total Revenue": None, "Net Income": None,
                           "Operating Income": None},
        })
        result = audit_quarterly(data, as_of="2027-01-15")
        self.assertIn("incomplete_latest_period", checks(result, "NEW"))
        self.assertEqual(result["companies"]["NEW"]["latest_period"], "2026-09-30")

    def test_identity_uses_minority_interest_from_raw_statements(self):
        # Assets 500 = liabilities 300 + equity 150 + minority 50.
        data = company("MIN", QUARTERS[:2], **{
            d: {"Stockholders Equity": 150.0} for d in QUARTERS[:2]
        })
        with tempfile.TemporaryDirectory() as raw:
            pd.DataFrame({
                "Date": QUARTERS[:2], "Ticker": "MIN",
                "Total Equity Gross Minority Interest": [200.0, 200.0],
            }).to_csv(Path(raw) / "MIN_balance_sheet.csv", index=False)
            with_raw = audit_quarterly(data, as_of="2025-08-01", raw_dir=raw)
        without_raw = audit_quarterly(data, as_of="2025-08-01")

        identity = [f for f in with_raw["findings"]
                    if f["check"] == "balance_sheet_identity"]
        self.assertEqual(identity, [])
        notes = [f for f in without_raw["findings"]
                 if f["check"] == "balance_sheet_identity"]
        self.assertTrue(notes and all(f["severity"] == "info" for f in notes))

    def test_cross_company_currency_and_calendar(self):
        data = pd.concat([
            company("AAA", QUARTERS),
            company("BBB.NS", QUARTERS),
            company("CCC", ["2025-04-30", "2025-07-31", "2025-10-31"]),
        ])
        result = audit_quarterly(data, as_of="2027-01-15",
                                 currencies={"AAA": "USD"})
        found = checks(result)
        self.assertIn("mixed_currencies", found)
        self.assertIn("mixed_fiscal_calendars", found)
        self.assertEqual(
            result["companies"]["BBB.NS"]["reporting_currency"]["currency"], "INR"
        )


if __name__ == "__main__":
    unittest.main()
