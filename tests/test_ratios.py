"""Ratio formulas checked against hand-calculated values.

Run: python -m unittest discover -s tests -t .
"""

import math
import unittest

import pandas as pd

from src.data_quality.verify_features import (
    check_feature_timing,
    hand_ratio,
    verify_ratio_formulas,
    verify_target,
)
from src.ml.prepare_revenue_decline_features import add_ratio_features


# Apple, quarter ended 2025-06-30 (values in USD, from the committed dataset).
AAPL_Q = {
    "Date": "2025-06-30",
    "Ticker": "AAPL",
    "Total Revenue": 94_036_000_000,
    "Net Income": 23_434_000_000,
    "Operating Income": 28_202_000_000,
    "Total Assets": 331_495_000_000,
    "Total Debt": 101_698_000_000,
    "Stockholders Equity": 65_830_000_000,
    "Current Assets": 122_491_000_000,
    "Current Liabilities": 141_120_000_000,
}

# Worked by hand (calculator), rounded to 6 decimals:
#   23,434 / 94,036   = 0.249202
#   28,202 / 94,036   = 0.299906
#  101,698 / 331,495  = 0.306786
#  101,698 / 65,830   = 1.544858
#  122,491 / 141,120  = 0.867992
AAPL_EXPECTED = {
    "Net_Income_Margin": 0.249202,
    "Operating_Margin": 0.299906,
    "Debt_To_Assets": 0.306786,
    "Debt_To_Equity": 1.544858,
    "Current_Ratio": 0.867992,
}


class RatioFormulaTests(unittest.TestCase):
    def test_known_values(self):
        row = add_ratio_features(pd.DataFrame([AAPL_Q])).iloc[0]
        for feature, expected in AAPL_EXPECTED.items():
            self.assertAlmostEqual(row[feature], expected, places=6, msg=feature)

    def test_zero_or_negative_denominators_give_missing(self):
        data = pd.DataFrame([
            dict(AAPL_Q, **{"Total Revenue": 0}),
            dict(AAPL_Q, **{"Total Revenue": -5.0}),
            dict(AAPL_Q, **{"Stockholders Equity": -1_000.0}),
            dict(AAPL_Q, **{"Current Liabilities": 0}),
            dict(AAPL_Q, **{"Total Assets": float("nan")}),
        ])
        result = add_ratio_features(data)
        self.assertTrue(math.isnan(result.loc[0, "Net_Income_Margin"]))
        self.assertTrue(math.isnan(result.loc[0, "Operating_Margin"]))
        self.assertTrue(math.isnan(result.loc[1, "Net_Income_Margin"]))
        self.assertTrue(math.isnan(result.loc[2, "Debt_To_Equity"]))
        self.assertTrue(math.isnan(result.loc[3, "Current_Ratio"]))
        self.assertTrue(math.isnan(result.loc[4, "Debt_To_Assets"]))
        # Unaffected ratios are still calculated.
        self.assertAlmostEqual(result.loc[2, "Debt_To_Assets"], 0.306786, places=6)

    def test_missing_source_column_skips_ratio(self):
        data = pd.DataFrame([{k: v for k, v in AAPL_Q.items()
                              if k != "Current Assets"}])
        self.assertNotIn("Current_Ratio", add_ratio_features(data).columns)

    def test_input_is_not_modified(self):
        data = pd.DataFrame([AAPL_Q])
        add_ratio_features(data)
        self.assertNotIn("Net_Income_Margin", data.columns)

    def test_reference_implementation(self):
        self.assertAlmostEqual(hand_ratio(1, 4), 0.25)
        self.assertIsNone(hand_ratio(1, 0))
        self.assertIsNone(hand_ratio(1, -2))
        self.assertIsNone(hand_ratio(None, 2))
        self.assertIsNone(hand_ratio(float("nan"), 2))

    def test_verifier_detects_wrong_stored_value(self):
        data = add_ratio_features(pd.DataFrame([AAPL_Q]))
        data.loc[0, "Current_Ratio"] = 1.5
        result = verify_ratio_formulas(data)
        self.assertEqual(result["features"]["Current_Ratio"]["status"], "mismatch")
        self.assertEqual(result["features"]["Net_Income_Margin"]["status"], "ok")


class LeakageAndTargetTests(unittest.TestCase):
    def test_future_columns_are_flagged(self):
        problems = check_feature_timing({
            "ok": ["Net_Income_Margin", "Current_Ratio"],
            "bad": ["Next_Revenue", "Ticker", "Mystery_Feature"],
        })
        self.assertEqual(len(problems), 3)
        self.assertTrue(all(p.startswith("bad:") for p in problems))

    def test_target_recomputation(self):
        base = pd.DataFrame({
            "Ticker": ["X"] * 4,
            "Date": ["2025-03-31", "2025-06-30", "2025-09-30", "2026-03-31"],
            "Total Revenue": [100, 90, 95, 80],
        })
        # 03-31 -> 06-30: decline (1); 06-30 -> 09-30: growth (0);
        # 09-30 -> 03-31 is 182 days apart, so it is not labelled.
        labelled = pd.DataFrame({
            "Ticker": ["X", "X"],
            "Date": ["2025-03-31", "2025-06-30"],
            "Next_Period_Revenue_Decline": [1, 0],
        })
        result = verify_target(base, labelled)
        self.assertEqual(result["expected_labels"], 2)
        self.assertEqual(result["disagreements"], [])

        labelled.loc[1, "Next_Period_Revenue_Decline"] = 1
        self.assertEqual(len(verify_target(base, labelled)["disagreements"]), 1)


if __name__ == "__main__":
    unittest.main()
