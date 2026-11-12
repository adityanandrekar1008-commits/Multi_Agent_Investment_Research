"""Distress features, labels, folds and metrics (offline, synthetic data).

Run: python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np
import pandas as pd

from src.ml.distress import model as dm
from src.ml.distress.features import (
    FEATURES, TARGET, add_features, add_labels, build_training_set, yahoo_annual_panel,
)


def company_years(cik, years, overrides=None):
    rows = []
    for i, year in enumerate(years):
        row = {"cik": cik, "company": f"C{cik}", "fiscal_year_end": f"{year}-12-31",
               "revenue": 100.0, "net_income": 10.0, "operating_income": 15.0,
               "interest_expense": 2.0, "operating_cash_flow": 12.0, "gross_profit": 40.0,
               "total_assets": 200e6, "total_liabilities": 100e6, "equity": 100e6,
               "current_assets": 80.0, "current_liabilities": 40.0,
               "retained_earnings": 50e6, "cash": 20e6, "long_term_debt": 30e6}
        row.update((overrides or {}).get(year, {}))
        rows.append(row)
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    def test_distress_definition_and_next_year_label(self):
        panel = company_years(1, [2015, 2016, 2017, 2018], {
            2016: {"net_income": -5.0, "operating_cash_flow": -1.0},      # cash-burning loss
            2017: {"equity": -1.0, "total_liabilities": 201e6},            # negative equity
        })
        p = add_labels(add_features(panel))
        self.assertEqual(p["distress_now"].tolist(), [0.0, 1.0, 1.0, 0.0])
        self.assertEqual(p[TARGET].tolist()[:3], [1.0, 1.0, 0.0])
        self.assertTrue(pd.isna(p[TARGET].iloc[3]))          # no next year
        self.assertEqual(p["cash_burn_now"].iloc[1], 1.0)
        self.assertEqual(p["negative_equity_now"].iloc[2], 1.0)

    def test_weak_interest_coverage(self):
        panel = company_years(2, [2015, 2016], {
            2016: {"operating_income": 1e6, "interest_expense": 2e6}})   # coverage 0.5, material
        p = add_features(panel)
        self.assertEqual(p["weak_coverage_now"].tolist(), [0.0, 1.0])
        self.assertEqual(p["distress_now"].iloc[1], 1.0)

    def test_gap_years_do_not_get_labels_or_changes(self):
        panel = company_years(3, [2015, 2017])                 # 2016 missing
        p = add_labels(add_features(panel))
        self.assertTrue(pd.isna(p[TARGET].iloc[0]))
        self.assertTrue(pd.isna(p["revenue_growth"].iloc[1]))

    def test_altman_and_ratios(self):
        p = add_features(company_years(4, [2015]))
        row = p.iloc[0]
        self.assertAlmostEqual(row["leverage"], 0.5)
        self.assertAlmostEqual(row["current_ratio"], 2.0)
        expected = (6.56 * (40 / 200e6) + 3.26 * 0.25 + 6.72 * (15 / 200e6) + 1.05 * 1.0)
        self.assertAlmostEqual(row["altman_z2"], expected)
        self.assertEqual(set(FEATURES) - set(p.columns), set())

    def test_training_set_filters(self):
        panel = pd.concat([
            company_years(5, [2015, 2016]),
            company_years(6, [2015, 2016], {2015: {"current_assets": np.nan}}),   # bank-like
            company_years(7, [2015, 2016], {2015: {"total_assets": 1e6}}),        # shell
        ])
        ciks = set(build_training_set(panel)["cik"])
        self.assertEqual(ciks, {5})

    def test_yahoo_mapping(self):
        dates = [pd.Timestamp("2024-03-31"), pd.Timestamp("2025-03-31")]
        income = pd.DataFrame({d: {"Total Revenue": 100.0, "Net Income": 5.0,
                                   "Operating Income": 8.0, "Interest Expense": -2.0}
                               for d in dates})
        balance = pd.DataFrame({d: {"Total Assets": 300.0, "Stockholders Equity": 120.0,
                                    "Current Assets": 90.0, "Current Liabilities": 60.0}
                                for d in dates})
        panel = yahoo_annual_panel(income, balance, pd.DataFrame(), key="X")
        self.assertEqual(len(panel), 2)
        self.assertEqual(panel["interest_expense"].tolist(), [2.0, 2.0])   # sign removed
        p = add_features(panel)
        self.assertAlmostEqual(p["leverage"].iloc[0], (300 - 120) / 300)  # derived liabilities


class ModelPlumbingTests(unittest.TestCase):
    def frame(self):
        rows = []
        for cik in range(20):
            for year in range(2010, 2020):
                rows.append({"cik": cik, "year": year,
                             "available_date": pd.Timestamp(f"{year + 1}-03-31"),
                             "label_available_date": pd.Timestamp(f"{year + 2}-03-31")})
        return pd.DataFrame(rows)

    def test_walk_forward_uses_only_known_labels(self):
        df = self.frame()
        for scheme in ("walk_forward", "walk_forward_unseen_company"):
            for train, test in dm.make_folds(df, scheme):
                cutoff = df.loc[test, "available_date"].min()
                self.assertTrue((df.loc[train, "label_available_date"] <= cutoff).all())
                if scheme == "walk_forward_unseen_company":
                    self.assertTrue(set(df.loc[train, "cik"]).isdisjoint(df.loc[test, "cik"]))

    def test_unseen_company_folds_partition_companies(self):
        df = self.frame()
        folds = dm.make_folds(df, "unseen_company")
        self.assertEqual(len(folds), dm.N_FOLDS)
        tested = np.concatenate([t for _, t in folds])
        self.assertEqual(sorted(tested), list(range(len(df))))

    def test_metrics(self):
        y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
        s = np.array([.1, .2, .3, .9, .8, .4, .7, .2, .1, .3])
        m = dm.metrics(y, s)
        self.assertEqual(m["roc_auc"], 1.0)
        self.assertEqual(m["top10_precision"], 1.0)
        self.assertAlmostEqual(m["top10_recall"], 1 / 3)
        self.assertEqual(m["balanced_accuracy"], 1.0)

    def test_cluster_bootstrap_ci_contains_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        s = y + rng.normal(0, 0.8, 400)
        groups = np.repeat(np.arange(100), 4)
        ci = dm.cluster_bootstrap(y, {"m": s}, groups, n=100)["m"]
        from sklearn.metrics import roc_auc_score
        self.assertLess(ci[0], roc_auc_score(y, s))
        self.assertGreater(ci[1], roc_auc_score(y, s))


if __name__ == "__main__":
    unittest.main()
