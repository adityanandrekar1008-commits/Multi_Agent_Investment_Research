"""Rule-based risk flags.

Run: python -m unittest discover -s tests -t .
"""

import unittest

import pandas as pd

from src.pipeline.risk_flags import evaluate_risk_flags


def quarter(date, **overrides):
    row = {
        "Date": date, "Total Revenue": 100.0, "Net Income": 10.0,
        "Operating Income": 15.0, "EBIT": 15.0, "Interest Expense": 2.0,
        "Total Assets": 500.0, "Total Debt": 100.0, "Stockholders Equity": 200.0,
        "Current Assets": 150.0, "Current Liabilities": 100.0,
        "Operating Cash Flow": 20.0, "Free Cash Flow": 12.0,
    }
    row.update(overrides)
    return row


DATES = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]


def flags_by_id(table, **kwargs):
    result = evaluate_risk_flags(pd.DataFrame(table), **kwargs)
    return result, {f["id"]: f for f in result["flags"]}


class RiskFlagTests(unittest.TestCase):
    def test_healthy_company_triggers_nothing(self):
        result, flags = flags_by_id([quarter(d) for d in DATES])
        self.assertEqual(result["summary"]["triggered"], 0)
        self.assertEqual(result["summary"]["not_evaluable"], 0)
        self.assertEqual(flags["revenue_decline_yoy"]["value"], 0.0)

    def test_every_rule_can_trigger(self):
        rows = [quarter(d) for d in DATES[:-1]]
        rows.append(quarter(
            DATES[-1], **{
                "Total Revenue": 80.0, "Net Income": -5.0, "Operating Income": -2.0,
                "EBIT": 1.0, "Interest Expense": 2.0, "Total Debt": 300.0,
                "Stockholders Equity": -10.0, "Current Assets": 50.0,
                "Operating Cash Flow": -1.0, "Free Cash Flow": -3.0,
            }))
        result, flags = flags_by_id(rows)
        for rule in ["net_loss", "operating_loss", "revenue_decline_yoy",
                     "operating_margin_drop_yoy", "low_current_ratio",
                     "negative_equity", "high_debt_to_assets",
                     "low_interest_coverage", "negative_operating_cash_flow",
                     "negative_free_cash_flow"]:
            self.assertEqual(flags[rule]["status"], "triggered", rule)
        # Negative equity makes debt/equity meaningless rather than "high".
        self.assertEqual(flags["high_debt_to_equity"]["status"], "not_evaluable")
        self.assertAlmostEqual(flags["revenue_decline_yoy"]["value"], -0.2)
        self.assertEqual(result["summary"]["triggered_by_severity"]["high"], 1)

    def test_threshold_edges(self):
        rows = [quarter(d) for d in DATES[:-1]]
        rows.append(quarter(DATES[-1], **{
            "Current Assets": 100.0, "Total Debt": 250.0,   # CR = 1.0, D/A = 0.5
            "Stockholders Equity": 125.0,                   # D/E = 2.0
        }))
        _, flags = flags_by_id(rows)
        self.assertEqual(flags["low_current_ratio"]["status"], "not_triggered")
        self.assertEqual(flags["high_debt_to_assets"]["status"], "not_triggered")
        self.assertEqual(flags["high_debt_to_equity"]["status"], "not_triggered")

    def test_missing_data_is_not_evaluable_not_ok(self):
        rows = [quarter(d, **{"Operating Cash Flow": None, "Free Cash Flow": None})
                for d in DATES[-2:]]  # no year-earlier quarter either
        _, flags = flags_by_id(rows)
        for rule in ["negative_operating_cash_flow", "negative_free_cash_flow",
                     "revenue_decline_yoy", "operating_margin_drop_yoy"]:
            self.assertEqual(flags[rule]["status"], "not_evaluable", rule)
            self.assertTrue(flags[rule]["explanation"])

    def test_uses_latest_quarter_with_inputs_and_says_so(self):
        rows = [quarter(d) for d in DATES[:-1]]
        rows.append(quarter(DATES[-1], **{"Current Assets": None,
                                          "Current Liabilities": None}))
        _, flags = flags_by_id(rows)
        flag = flags["low_current_ratio"]
        self.assertEqual(flag["period_end"], "2025-12-31")
        self.assertIn("latest quarter with the required figures", flag["explanation"])

    def test_bank_rules_not_applicable(self):
        _, flags = flags_by_id([quarter(d) for d in DATES], financial_company=True)
        for rule in ["operating_loss", "operating_margin_drop_yoy", "low_current_ratio"]:
            self.assertEqual(flags[rule]["status"], "not_evaluable", rule)
        self.assertEqual(flags["net_loss"]["status"], "not_triggered")


if __name__ == "__main__":
    unittest.main()
