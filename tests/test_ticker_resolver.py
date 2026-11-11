"""Ticker/company resolution with a fake provider (no network).

Run: python -m unittest discover -s tests -t .
"""

import unittest

from src.data_collection.provider import ProviderError, _looks_like_failure
from src.data_collection.ticker_resolver import TickerResolutionError, resolve


class FakeProvider:
    """Symbols in `with_data` have quarterly statements; `search_results`
    maps a query to provider quotes."""

    name = "fake"

    def __init__(self, with_data=(), search_results=None, fail=False):
        self.with_data = set(with_data)
        self.search_results = search_results or {}
        self.fail = fail
        self.checked = []

    def has_quarterly_financials(self, symbol):
        if self.fail:
            raise ProviderError("network down")
        self.checked.append(symbol)
        return symbol in self.with_data

    def search(self, query, limit=10):
        if self.fail:
            raise ProviderError("network down")
        return self.search_results.get(query, [])


def quote(symbol, name, exchange, quote_type="EQUITY"):
    return {"symbol": symbol, "shortname": name, "exchange": exchange,
            "quoteType": quote_type}


class ResolverTests(unittest.TestCase):
    def test_exchange_qualified_ticker_used_exactly(self):
        provider = FakeProvider(with_data={"RELIANCE.NS"})
        result = resolve("reliance.ns", provider)
        self.assertEqual(result.symbol, "RELIANCE.NS")
        self.assertEqual(result.method, "exact")
        self.assertEqual(provider.checked, ["RELIANCE.NS"])

    def test_bare_us_symbol_beats_unrelated_indian_listing(self):
        # META.BO is String Metaverse Ltd, not Meta Platforms.
        provider = FakeProvider(with_data={"META", "META.BO"})
        result = resolve("META", provider)
        self.assertEqual(result.symbol, "META")
        self.assertEqual(result.method, "as_typed")
        self.assertEqual([a["symbol"] for a in result.alternatives], ["META.BO"])

    def test_bare_indian_symbol_gets_nse_suffix(self):
        provider = FakeProvider(with_data={"RELIANCE.NS", "RELIANCE.BO"})
        result = resolve("Reliance", provider)
        self.assertEqual(result.symbol, "RELIANCE.NS")
        self.assertEqual(result.method, "suffix")

    def test_prefer_indian_option(self):
        provider = FakeProvider(with_data={"INFY", "INFY.NS"})
        self.assertEqual(resolve("INFY", provider).symbol, "INFY")
        self.assertEqual(
            resolve("INFY", provider, prefer_indian=True).symbol, "INFY.NS"
        )

    def test_company_name_search(self):
        provider = FakeProvider(
            with_data={"RELIANCE.NS"},
            search_results={"Reliance Industries": [
                quote("RIGD.IL", "Reliance GDR", "IOB"),
                quote("RELIANCE.NS", "RELIANCE INDUSTRIES LTD", "NSI"),
                quote("RELI-ETF", "Some fund", "NYQ", quote_type="ETF"),
            ]},
        )
        result = resolve("  Reliance   Industries ", provider)
        self.assertEqual(result.symbol, "RELIANCE.NS")
        self.assertEqual(result.method, "name_search")
        self.assertTrue(result.notes)

    def test_unknown_company(self):
        with self.assertRaises(TickerResolutionError) as ctx:
            resolve("ZZZQX", FakeProvider())
        self.assertEqual(ctx.exception.code, "unknown_company")

    def test_etf_is_not_a_company(self):
        provider = FakeProvider(search_results={"SPY": [
            quote("SPY", "SPDR S&P 500 ETF", "PCX", quote_type="ETF"),
        ]})
        with self.assertRaises(TickerResolutionError) as ctx:
            resolve("SPY", provider)
        self.assertEqual(ctx.exception.code, "not_a_company")
        self.assertEqual(ctx.exception.candidates[0]["type"], "ETF")

    def test_symbol_without_financials(self):
        provider = FakeProvider(search_results={"SPY.XX": []})
        with self.assertRaises(TickerResolutionError) as ctx:
            resolve("SPY.XX", provider)
        self.assertEqual(ctx.exception.code, "no_financial_data")

    def test_empty_input(self):
        with self.assertRaises(TickerResolutionError) as ctx:
            resolve("   ", FakeProvider())
        self.assertEqual(ctx.exception.code, "empty_input")

    def test_provider_failure_is_not_reported_as_unknown_ticker(self):
        with self.assertRaises(ProviderError):
            resolve("AAPL", FakeProvider(fail=True))


class ProviderFailureClassificationTests(unittest.TestCase):
    def test_network_warning_is_a_failure(self):
        # Exact warning yfinance 1.7 logs when the network is down.
        self.assertTrue(_looks_like_failure([
            "Cookie/crumb fetch failed (ConnectionError), continuing without crumb"
        ]))
        self.assertTrue(_looks_like_failure(["YFRateLimitError: Too Many Requests"]))

    def test_unrelated_message_is_not_a_failure(self):
        self.assertFalse(_looks_like_failure(["Some field was not reported"]))


if __name__ == "__main__":
    unittest.main()
