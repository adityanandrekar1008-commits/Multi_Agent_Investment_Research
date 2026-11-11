"""Splits, baselines and the evaluation's statistics.

Run: python -m unittest discover -s tests -t .
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.ml import evaluate_revenue_decline as ev
from src.ml.splits import date_boundary_split


QUARTER_ENDS = pd.to_datetime([
    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
    "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31",
    "2026-03-31", "2026-06-30",
])


def synthetic(signal, seed=0, companies=6):
    """Feature rows for several companies. With signal=True the net income
    margin separates the classes; otherwise every feature is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(companies):
        for date in QUARTER_ENDS:
            label = int(rng.random() < 0.4)
            margin = (-0.1 if label else 0.2) if signal else rng.normal(0.1, 0.1)
            rows.append({
                "Date": date, "Ticker": f"C{c}",
                ev.TARGET: label,
                "Net_Income_Margin": margin + rng.normal(0, 0.01),
                "Operating_Margin": rng.normal(0.2, 0.05),
                "Debt_To_Assets": rng.normal(0.3, 0.05),
                "Debt_To_Equity": rng.normal(1.0, 0.2),
                "Current_Ratio": rng.normal(1.5, 0.2),
            })
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def test_date_boundary_split_never_shares_a_date(self):
        df = synthetic(signal=False)
        train, test, cutoff = date_boundary_split(df, 0.7)
        self.assertTrue(set(train["Date"]).isdisjoint(set(test["Date"])))
        self.assertTrue((train["Date"] < cutoff).all())
        self.assertTrue((test["Date"] >= cutoff).all())
        self.assertGreaterEqual(len(train), 0.7 * len(df))
        self.assertEqual(len(train) + len(test), len(df))

    def test_date_boundary_split_needs_two_dates(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2025-03-31"] * 3)})
        with self.assertRaises(ValueError):
            date_boundary_split(df)

    def test_walk_forward_uses_only_observable_labels(self):
        df = synthetic(signal=False)
        df["Next_Date"] = df["Date"] + pd.Timedelta(days=91)
        for unseen in (False, True):
            for train_index, test_index in ev.walk_forward_folds(df, unseen):
                test_date = df.loc[test_index, "Date"].iloc[0]
                self.assertTrue((df.loc[train_index, "Next_Date"] <= test_date).all())
                if unseen:
                    ticker = df.loc[test_index[0], "Ticker"]
                    self.assertNotIn(ticker, set(df.loc[train_index, "Ticker"]))

    def test_leave_company_out(self):
        df = synthetic(signal=False)
        folds = ev.leave_company_out_folds(df)
        self.assertEqual(len(folds), df["Ticker"].nunique())
        for train_index, test_index in folds:
            self.assertTrue(set(df.loc[train_index, "Ticker"]).isdisjoint(
                set(df.loc[test_index, "Ticker"])))


class BaselineAndMetricTests(unittest.TestCase):
    def test_seasonal_baseline(self):
        X = np.array([[12], [12], [12], [3], [3], [6]])
        y = np.array([1, 1, 0, 0, 0, 0])
        model = ev.SeasonalBaseline().fit(X, y)
        self.assertEqual(model.predict(np.array([[12], [3], [9]])).tolist(), [1, 0, 0])
        self.assertAlmostEqual(model.predict_proba(np.array([[12]]))[0, 1], 2 / 3)

    def test_fast_balanced_accuracy_matches_sklearn(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            y = rng.integers(0, 2, 20)
            p = rng.integers(0, 2, 20)
            if len(set(y)) < 2:
                continue
            self.assertAlmostEqual(
                ev.fast_balanced_accuracy(y, p), balanced_accuracy_score(y, p)
            )
        self.assertIsNone(ev.fast_balanced_accuracy([1, 1], [1, 0]))

    def test_metrics_counts(self):
        m = ev.compute_metrics([1, 0, 1, 0, 0], [1, 1, 0, 0, 0])
        self.assertEqual(m["confusion_matrix"], {"tn": 2, "fp": 1, "fn": 1, "tp": 1})
        self.assertAlmostEqual(m["precision_decline"], 0.5)
        self.assertAlmostEqual(m["recall_decline"], 0.5)


class EndToEndTests(unittest.TestCase):
    def run_eval(self, df, scheme):
        with tempfile.TemporaryDirectory() as tmp:
            features = Path(tmp) / "features.csv"
            df.to_csv(features, index=False)
            results = ev.evaluate(
                features_file=features, target_file=Path(tmp) / "missing.csv",
                output_dir=Path(tmp) / "out", n_permutations=30,
                n_bootstrap=200, n_jobs=1, schemes=[scheme],
                echo=lambda *a: None,
            )
            for name in ev.OUTPUT_NAMES.values():
                self.assertTrue((Path(tmp) / "out" / name).exists())
            with self.assertRaises(FileExistsError):
                ev.evaluate(features_file=features, output_dir=Path(tmp) / "out",
                            n_permutations=0, echo=lambda *a: None)
        return results

    def test_real_signal_is_detected(self):
        results = self.run_eval(synthetic(signal=True), "walk_forward")
        model = results["schemes"]["walk_forward"]["models"]["logreg_5_ratios"]
        self.assertGreater(model["balanced_accuracy"], 0.9)
        self.assertLess(model["permutation_p_value"], 0.05)
        self.assertTrue(any("approximated" in n for n in results["notes"]))

    def test_noise_is_not_called_signal(self):
        results = self.run_eval(synthetic(signal=False, seed=3), "unseen_company")
        model = results["schemes"]["unseen_company"]["models"]["logreg_5_ratios"]
        self.assertGreater(model["permutation_p_value"], 0.05)
        self.assertIn("no evidence", results["conclusions"][-1])


if __name__ == "__main__":
    unittest.main()
