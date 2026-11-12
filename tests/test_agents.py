"""Market, News and Manager agents, the LangGraph flow and the HTTP API
(offline: fake provider, no LLM).

Run: python -m unittest discover -s tests -t .
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd

os.environ["AGENTS_USE_LLM"] = "0"

from src.agents import manager, market, news  # noqa: E402
from src.agents.common import result, signal  # noqa: E402
from src.pipeline.schema_check import load_schema, validate  # noqa: E402
from tests.test_analyze_offline import RAW, FakeProvider  # noqa: E402

REPORT_SCHEMA = load_schema(
    os.path.join(os.path.dirname(__file__), "..", "schemas", "research_report.schema.json"))


def prices(daily_returns, start="2024-01-01"):
    index = pd.bdate_range(start, periods=len(daily_returns) + 1)
    close = 100 * np.cumprod(np.r_[1.0, 1 + np.asarray(daily_returns)])
    return pd.DataFrame({"Close": close}, index=index)


class AgentProvider(FakeProvider):
    def __init__(self, stock_returns=None, news_items=None, **kwargs):
        super().__init__(**kwargs)
        rng = np.random.default_rng(1)
        self.index_returns = rng.normal(0.0004, 0.01, 520)
        self.stock_returns = stock_returns if stock_returns is not None else \
            1.5 * self.index_returns + rng.normal(0, 0.01, 520)
        self.news_items = news_items or []

    def price_history(self, symbol, period="2y"):
        r = self.index_returns if symbol.startswith("^") else self.stock_returns
        return prices(r)

    def news(self, symbol):
        return self.news_items

    def annual_statements(self, symbol):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def yahoo_item(title, days_ago=1):
    when = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {"content": {"title": title, "summary": "", "pubDate": when,
                        "provider": {"displayName": "Wire"},
                        "canonicalUrl": {"url": "https://example.com/" + title[:5]}}}


class MarketTests(unittest.TestCase):
    def test_metrics_and_signals(self):
        p = AgentProvider()
        m = market.compute_metrics(p.price_history("X"), p.price_history("^GSPC"))
        self.assertAlmostEqual(m["beta_1y"], 1.5, delta=0.25)
        self.assertGreater(m["volatility_1y"], 0.2)
        self.assertLessEqual(m["max_drawdown_1y"], 0)
        self.assertEqual(market.benchmark_for("TCS.NS"), "^NSEI")
        self.assertEqual(market.benchmark_for("AAPL"), "^GSPC")

    def test_crash_triggers_high_drawdown(self):
        r = np.r_[np.full(300, 0.001), np.full(60, -0.012), np.full(160, 0.0)]
        p = AgentProvider(stock_returns=r)
        out = market.run("X", p, {})
        ids = {s["id"] for s in out["signals"]}
        self.assertIn("deep_drawdown", ids)
        self.assertEqual(out["status"], "ok")

    def test_too_little_history(self):
        p = AgentProvider(stock_returns=np.zeros(20))
        self.assertEqual(market.run("X", p, {})["status"], "unavailable")


class NewsTests(unittest.TestCase):
    def test_lexicon_events(self):
        c = news.lexicon_classify({"title": "Acme files for Chapter 11 bankruptcy", "summary": ""})
        self.assertEqual(c["event"], "bankruptcy_or_default")
        self.assertLess(c["sentiment"], 0)
        c = news.lexicon_classify({"title": "Acme beats estimates, raises guidance", "summary": ""})
        self.assertEqual(c["event"], "beat_or_raise")
        self.assertGreater(c["sentiment"], 0)
        c = news.lexicon_classify({"title": "Acme hits 18-month low", "summary": ""})
        self.assertLess(c["sentiment"], 0)

    def test_search_name(self):
        self.assertEqual(news.search_name("Reliance Industries Limited"), "Reliance Industries")
        self.assertEqual(news.search_name("Apple Inc."), "Apple")

    def test_run_dedupes_and_flags_events(self):
        items = [yahoo_item("Regulator opens probe into Acme accounts"),
                 yahoo_item("Regulator opens probe into Acme accounts"),        # duplicate
                 yahoo_item("Acme announces layoffs of 2,000 jobs", 2),
                 yahoo_item("Old story about Acme", 400)]                        # too old
        out = news.run("ACME", "Acme Corp", AgentProvider(news_items=items), use_google=False)
        self.assertEqual(out["data"]["articles"], 2)
        ids = {s["id"] for s in out["signals"]}
        self.assertIn("news_legal_or_regulatory", ids)
        self.assertIn("news_layoffs", ids)
        self.assertIn("low_news_coverage", ids)
        self.assertEqual(out["data"]["method"], "lexicon")

    def test_no_news(self):
        out = news.run("ACME", "Acme", AgentProvider(), use_google=False)
        self.assertEqual(out["status"], "unavailable")


class ManagerLogicTests(unittest.TestCase):
    def agents(self, quant=(), mkt=(), nws=()):
        mk = lambda name, sigs: result(name, "ok", f"{name} summary", list(sigs))  # noqa: E731
        return {"quantitative_risk": mk("quantitative_risk", quant),
                "market": mk("market", mkt), "news": mk("news", nws)}

    def test_attention_levels(self):
        neg = lambda sev, sid="x": signal(sid, sev, "negative", "t", "d")  # noqa: E731
        self.assertEqual(manager.attention_level(self.agents()), "low")
        self.assertEqual(manager.attention_level(self.agents(mkt=[neg("medium")])), "moderate")
        self.assertEqual(manager.attention_level(
            self.agents(quant=[neg("medium")], mkt=[neg("medium")], nws=[neg("medium")])), "elevated")
        self.assertEqual(manager.attention_level(self.agents(mkt=[neg("high")])), "elevated")
        self.assertEqual(manager.attention_level(
            self.agents(quant=[neg("high")], mkt=[neg("high")])), "high")
        self.assertEqual(manager.attention_level(
            self.agents(quant=[neg("high", "distress_now")])), "high")

    def test_cross_checks(self):
        neg = signal("x", "high", "negative", "t", "d")
        notes = manager.cross_checks(self.agents(mkt=[neg]))
        self.assertEqual(notes[0]["type"], "conflict")
        notes = manager.cross_checks(self.agents(quant=[neg], mkt=[neg]))
        self.assertIn("agreement", {n["type"] for n in notes})
        notes = manager.cross_checks(self.agents())
        self.assertEqual(notes[0]["type"], "agreement")


@unittest.skipUnless((RAW / "AAPL_income_statement.csv").exists(), "committed data needed")
class GraphTests(unittest.TestCase):
    def test_full_graph_report_is_schema_valid(self):
        provider = AgentProvider(news_items=[yahoo_item("Apple beats estimates")])
        with tempfile.TemporaryDirectory() as tmp:
            report = manager.run_research("AAPL", provider=provider, output_root=tmp,
                                          use_google_news=False, distress_bundle=None)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(validate(report, REPORT_SCHEMA), [])
        self.assertEqual(set(report["agents"]), {"quantitative_risk", "market", "news"})
        self.assertEqual(report["manager"]["summary_source"], "template")
        self.assertIn(report["manager"]["attention_level"], ("low", "moderate", "elevated", "high"))
        self.assertEqual(report["analysis"]["schema_version"], "company_analysis/1.0")

    def test_unknown_company_stops_after_resolve(self):
        report = manager.run_research("ZZZQXNOTREAL", provider=AgentProvider(),
                                      use_google_news=False)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["code"], "unknown_company")
        self.assertEqual(validate(report, REPORT_SCHEMA), [])

    def test_failing_agent_does_not_break_report(self):
        provider = AgentProvider()
        with mock.patch.object(market, "run", side_effect=RuntimeError("boom")), \
                tempfile.TemporaryDirectory() as tmp:
            report = manager.run_research("AAPL", provider=provider, output_root=tmp,
                                          use_google_news=False)
        self.assertEqual(report["agents"]["market"]["status"], "error")
        self.assertEqual(validate(report, REPORT_SCHEMA), [])


@unittest.skipUnless((RAW / "AAPL_income_statement.csv").exists(), "committed data needed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        self.tmp = tempfile.TemporaryDirectory()
        app = create_app(output_root=self.tmp.name, use_google_news=False,
                         provider_factory=lambda: AgentProvider())
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_endpoints(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        r = self.client.get("/api/analyze", params={"company": "AAPL"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["schema_version"], "company_analysis/1.0")
        r = self.client.get("/api/research", params={"company": "AAPL"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["schema_version"], "research_report/1.0")
        self.assertEqual(self.client.get("/api/research", params={"company": "ZZZQXNOTREAL"}).status_code, 404)
        self.assertEqual(self.client.get("/api/analyze", params={"company": " "}).status_code, 400)
        self.assertIn("text/html", self.client.get("/").headers["content-type"])


if __name__ == "__main__":
    unittest.main()


class ProviderCacheTests(unittest.TestCase):
    def test_annual_and_quarterly_statements_do_not_collide(self):
        from src.data_collection.provider import YahooFinanceProvider

        annual = pd.DataFrame({"2025": [1.0]}, index=["Total Assets"])
        quarterly = pd.DataFrame({"2026-06": [2.0]}, index=["Total Assets"])

        class FakeTicker:
            def __init__(self, symbol):
                self.quarterly_balance_sheet = quarterly
                self.balance_sheet = annual
                self.income_stmt = self.cashflow = pd.DataFrame()

        provider = YahooFinanceProvider.__new__(YahooFinanceProvider)
        provider._yf = mock.Mock(Ticker=FakeTicker)
        provider._statements, provider._profiles = {}, {}
        provider.quarterly_statement("X", "balance_sheet")        # cached first
        _, balance, _ = provider.annual_statements("X")
        self.assertIs(balance, annual)
