"""End-to-end `analyze` without network, plus the output schema.

A fake provider serves the committed AAPL quarterly statements in the same
layout yfinance returns, so the whole pipeline runs offline.

Run: python -m unittest discover -s tests -t .
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_collection.provider import ProviderError
from src.pipeline.analyze import analyze_company
from src.pipeline.schema_check import load_schema, validate, validate_payload

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "quarterly"
EXAMPLES = ROOT / "docs" / "examples"


class FakeProvider:
    name = "fake provider (committed data)"

    def __init__(self, fail_on=None, profile=None):
        self.fail_on = fail_on
        self._profile = profile if profile is not None else {
            "longName": "Apple Inc.", "exchange": "NMS", "currency": "USD",
            "financialCurrency": "USD", "sector": "Technology", "country": "United States",
        }

    def quarterly_statement(self, symbol, statement):
        if self.fail_on == "statement":
            raise ProviderError("simulated network failure")
        path = RAW / f"{symbol}_{statement}.csv"
        if not path.exists():
            return pd.DataFrame()
        rows = pd.read_csv(path)
        rows["Date"] = pd.to_datetime(rows["Date"])
        # yfinance layout: line items as rows, period ends as columns.
        return rows.drop(columns=["Ticker"]).set_index("Date").T

    def has_quarterly_financials(self, symbol):
        if self.fail_on == "resolve":
            raise ProviderError("simulated network failure")
        return not self.quarterly_statement(symbol, "income_statement").empty

    def search(self, query, limit=10):
        return []

    def profile(self, symbol):
        if self.fail_on == "profile":
            raise RuntimeError("simulated bug")
        return dict(self._profile)


@unittest.skipUnless((RAW / "AAPL_income_statement.csv").exists(),
                     "committed raw data not present")
class AnalyzeOfflineTests(unittest.TestCase):
    def run_analyze(self, text, **provider_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            payload = analyze_company(
                text, output_root=tmp, provider=FakeProvider(**provider_kwargs),
                as_of="2026-11-07", echo=lambda *a: None,
            )
            folder = next(Path(tmp).iterdir())
            files = {p.name for p in folder.iterdir()}
            saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        return payload, files, saved

    def test_full_run_is_schema_valid(self):
        payload, files, saved = self.run_analyze("AAPL")
        self.assertEqual(payload["status"], "ok", payload["error"])
        self.assertEqual(validate_payload(payload), [])
        self.assertEqual(saved["schema_version"], "company_analysis/1.0")
        for name in ["summary.json", "report.md", "quarterly_financials.csv",
                     "data_quality.md", "pipeline.log", "raw"]:
            self.assertIn(name, files)
        self.assertEqual(payload["company"]["reporting_currency"], "USD")
        self.assertEqual(len(payload["risk_flags"]["flags"]), 11)
        self.assertEqual(payload["ml_signal"]["label"], "experimental")
        # Deterministic and ML sections stay separate.
        self.assertNotIn("score", json.dumps(payload["risk_flags"]))

    def test_error_states_are_schema_valid(self):
        cases = [
            ("   ", {}, "empty_input"),
            ("ZZZQXNOTREAL", {}, "unknown_company"),
            ("AAPL", {"fail_on": "resolve"}, "provider_error"),
            ("AAPL", {"fail_on": "statement"}, "provider_error"),
            ("AAPL", {"fail_on": "profile"}, "internal_error"),
        ]
        for text, kwargs, code in cases:
            payload, files, _ = self.run_analyze(text, **kwargs)
            self.assertEqual(payload["status"], "error", code)
            self.assertEqual(payload["error"]["code"], code)
            self.assertEqual(validate_payload(payload), [], code)
            self.assertIn("summary.json", files)
            if code == "internal_error":
                self.assertIn("error.log", files)


class SchemaTests(unittest.TestCase):
    def test_examples_are_valid(self):
        examples = sorted(EXAMPLES.glob("*.json"))
        self.assertTrue(examples)
        for path in examples:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(validate_payload(payload), [], path.name)

    def test_validator_rejects_broken_payloads(self):
        schema = load_schema()
        good = json.loads((EXAMPLES / "company_analysis_ok_AAPL.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(validate(good, schema), [])

        def broken(change):
            payload = copy.deepcopy(good)
            change(payload)
            return validate(payload, schema)

        self.assertTrue(broken(lambda p: p.pop("disclaimer")))
        self.assertTrue(broken(lambda p: p.update(extra_field=1)))
        self.assertTrue(broken(lambda p: p.update(status="maybe")))
        self.assertTrue(broken(lambda p: p["risk_flags"]["flags"][0]
                               .update(status="fine")))
        self.assertTrue(broken(lambda p: p["ml_signal"].update(label="reliable")))
        self.assertTrue(broken(lambda p: p["metrics"]["latest"]["values"]
                               .update(revenue="lots")))
        # ok status with an error object is inconsistent.
        self.assertTrue(broken(lambda p: p.update(
            error={"code": "unknown_company", "message": "x", "candidates": []})))


if __name__ == "__main__":
    unittest.main()
