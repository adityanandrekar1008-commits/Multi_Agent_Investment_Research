"""Financial data provider wrapper (Yahoo Finance via yfinance).

yfinance usually returns an empty table both when a company has no data and
when the request failed (network error, rate limit). This wrapper captures
yfinance's warning/error log during each call so the pipeline can tell the
two apart:

- empty table, no provider error  -> the company has no data
- exception or logged error        -> ProviderError (try again later)

Statements are cached per symbol so ticker resolution and collection do not
download the same table twice. Tests can pass a fake provider with the same
methods.
"""

from __future__ import annotations

import logging

import pandas as pd


STATEMENT_ATTRIBUTES = {
    "income_statement": "quarterly_income_stmt",
    "balance_sheet": "quarterly_balance_sheet",
    "cash_flow": "quarterly_cashflow",
}


class ProviderError(RuntimeError):
    """The data provider could not be reached or refused the request."""

    code = "provider_error"


class _ErrorCapture(logging.Handler):
    # yfinance reports some network failures only as warnings, for example
    # "Cookie/crumb fetch failed (ConnectionError)", and then returns an
    # empty table. Capture warnings too.
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class YahooFinanceProvider:
    name = "Yahoo Finance (via yfinance)"

    def __init__(self):
        import yfinance  # imported lazily so tests do not need network access

        self._yf = yfinance
        self._statements = {}
        self._profiles = {}

    def _call(self, description, func):
        logger = logging.getLogger("yfinance")
        capture = _ErrorCapture()
        logger.addHandler(capture)
        try:
            result = func()
        except Exception as exc:  # network, HTTP, parsing errors
            raise ProviderError(f"{description} failed: {exc}") from exc
        finally:
            logger.removeHandler(capture)
        return result, capture.messages

    def quarterly_statement(self, symbol, statement):
        """Return the yfinance quarterly statement (line items x dates).

        Returns an empty DataFrame when the provider has no data for the
        symbol. Raises ProviderError when the request itself failed.
        """
        key = (symbol, statement)
        if key in self._statements:
            return self._statements[key]

        attribute = STATEMENT_ATTRIBUTES[statement]
        table, errors = self._call(
            f"Downloading {statement} for {symbol}",
            lambda: getattr(self._yf.Ticker(symbol), attribute),
        )
        if table is None:
            table = pd.DataFrame()

        if table.empty and errors and _looks_like_failure(errors):
            raise ProviderError(
                f"Downloading {statement} for {symbol} failed: {errors[-1]}"
            )

        self._statements[key] = table
        return table

    def has_quarterly_financials(self, symbol):
        return not self.quarterly_statement(symbol, "income_statement").empty

    def search(self, query, limit=10):
        """Search listed securities by company name or symbol."""
        result, errors = self._call(
            f"Searching for '{query}'",
            lambda: self._yf.Search(
                query, max_results=limit, news_count=0
            ).quotes,
        )
        if not result and errors and _looks_like_failure(errors):
            raise ProviderError(f"Searching for '{query}' failed: {errors[-1]}")
        return list(result or [])

    def price_history(self, symbol, period="2y"):
        """Daily prices (DataFrame with Close etc.); empty if none."""
        key = (symbol, "history", period)
        if key in self._statements:
            return self._statements[key]
        table, errors = self._call(
            f"Downloading prices for {symbol}",
            lambda: self._yf.Ticker(symbol).history(period=period, auto_adjust=True),
        )
        if (table is None or table.empty) and errors and _looks_like_failure(errors):
            raise ProviderError(f"Downloading prices for {symbol} failed: {errors[-1]}")
        table = pd.DataFrame() if table is None else table
        self._statements[key] = table
        return table

    def news(self, symbol):
        """Recent news items from Yahoo Finance (list of dicts); [] if none."""
        try:
            items, _ = self._call(f"Fetching news for {symbol}",
                                  lambda: self._yf.Ticker(symbol).news)
        except ProviderError:
            return []
        return list(items or [])

    def annual_statements(self, symbol):
        """(income, balance, cashflow) annual statements, line items x dates."""
        out = []
        for attribute in ("income_stmt", "balance_sheet", "cashflow"):
            key = (symbol, "annual", attribute)  # distinct from quarterly cache keys
            if key not in self._statements:
                table, _ = self._call(
                    f"Downloading annual {attribute} for {symbol}",
                    lambda a=attribute: getattr(self._yf.Ticker(symbol), a),
                )
                self._statements[key] = pd.DataFrame() if table is None else table
            out.append(self._statements[key])
        return tuple(out)

    def profile(self, symbol):
        """Best-effort company profile. Returns {} if it cannot be fetched;
        a missing profile never stops the pipeline."""
        if symbol in self._profiles:
            return self._profiles[symbol]
        try:
            info, _ = self._call(
                f"Fetching profile for {symbol}",
                lambda: self._yf.Ticker(symbol).info,
            )
        except ProviderError:
            info = {}
        keys = [
            "longName", "shortName", "exchange", "fullExchangeName",
            "quoteType", "currency", "financialCurrency", "sector",
            "industry", "country", "marketCap", "trailingPE", "forwardPE",
            "priceToBook", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        profile = {key: info.get(key) for key in keys if info and info.get(key)}
        self._profiles[symbol] = profile
        return profile


def _looks_like_failure(messages):
    """Heuristic: yfinance error messages that mean the request failed rather
    than 'this symbol has no data'."""
    failure_words = (
        "rate limit", "ratelimit", "too many requests", "timed out",
        "timeout", "connectionerror", "connection", "resolve host",
        "network", "ssl", "curl", "proxy", "unauthorized", "401", "403",
        "429", "500", "502", "503", "504",
    )
    text = " ".join(messages).lower()
    return any(word in text for word in failure_words)
