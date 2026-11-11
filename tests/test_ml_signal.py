"""Experimental ML signal.

Run: python -m unittest discover -s tests -t .
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.ml_signal import build_ml_signal

COMMITTED_TRAINING = Path(__file__).resolve().parents[1] / "data" / "processed" / \
    "revenue_decline_features.csv"


def training_file(directory):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        label = i % 3 == 0
        rows.append({
            "Date": f"2025-{(i % 12) + 1:02d}-28", "Ticker": f"C{i % 5}",
            "Next_Period_Revenue_Decline": int(label),
            "Net_Income_Margin": (0.05 if label else 0.2) + rng.normal(0, 0.02),
            "Operating_Margin": rng.normal(0.2, 0.05),
            "Debt_To_Assets": rng.normal(0.3, 0.05),
            "Debt_To_Equity": rng.normal(1.0, 0.2),
            "Current_Ratio": rng.normal(1.5, 0.2),
        })
    path = Path(directory) / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


LATEST = {"Net_Income_Margin": 0.1, "Operating_Margin": 0.2,
          "Debt_To_Assets": None, "Debt_To_Equity": 1.0, "Current_Ratio": 1.4}


class MlSignalTests(unittest.TestCase):
    def test_score_equals_intercept_plus_contributions(self):
        with tempfile.TemporaryDirectory() as tmp:
            signal = build_ml_signal(LATEST, "2026-03-31",
                                     training_file=training_file(tmp),
                                     evaluation_file=Path(tmp) / "none.json")
        self.assertEqual(signal["status"], "available")
        self.assertEqual(signal["label"], "experimental")
        logit = signal["intercept"] + sum(
            v["contribution"] for v in signal["inputs"].values())
        self.assertAlmostEqual(signal["score"], 1 / (1 + math.exp(-logit)), places=9)
        self.assertEqual(signal["predicted_class"],
                         "decline" if signal["score"] >= 0.5 else "no_decline")

    def test_missing_input_is_flagged_as_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            signal = build_ml_signal(LATEST, "2026-03-31",
                                     training_file=training_file(tmp),
                                     evaluation_file=Path(tmp) / "none.json")
        debt = signal["inputs"]["Debt_To_Assets"]
        self.assertTrue(debt["imputed"])
        self.assertIsNone(debt["value"])
        self.assertAlmostEqual(debt["value_used"], debt["training_median"])
        self.assertFalse(signal["inputs"]["Current_Ratio"]["imputed"])
        self.assertFalse(signal["evaluation"]["available"])

    def test_unavailable_cases(self):
        bank = build_ml_signal(LATEST, "2026-03-31", financial_company=True)
        self.assertEqual(bank["status"], "unavailable")
        empty = build_ml_signal({k: None for k in LATEST}, "2026-03-31")
        self.assertEqual(empty["status"], "unavailable")
        missing = build_ml_signal(LATEST, "2026-03-31",
                                  training_file=Path("does/not/exist.csv"))
        self.assertEqual(missing["status"], "unavailable")
        for signal in (bank, empty, missing):
            self.assertTrue(signal["reason"])
            self.assertTrue(signal["limitations"])

    @unittest.skipUnless(COMMITTED_TRAINING.exists(), "committed data not present")
    def test_runs_on_committed_training_data(self):
        signal = build_ml_signal(LATEST, "2026-03-31")
        self.assertEqual(signal["status"], "available")
        self.assertEqual(signal["training"]["rows"], 31)


if __name__ == "__main__":
    unittest.main()
