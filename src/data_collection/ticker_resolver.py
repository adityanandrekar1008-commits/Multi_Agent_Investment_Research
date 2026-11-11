"""Resolve user input (ticker or company name) to a provider symbol.

Rules
-----
- Exchange-qualified tickers (RELIANCE.NS, TCS.BO, 7203.T, BRK-B) are used
  exactly as typed.
- A bare symbol (RELIANCE, INFY, AAPL) is tried as typed first, then on NSE
  (.NS) and BSE (.BO). As-typed comes first because Indian exchanges reuse
  some US symbols for unrelated companies: META.BO is String Metaverse Ltd,
  not Meta Platforms. Bare Indian symbols such as RELIANCE or TCS have no
  as-typed listing, so they still resolve to .NS. If more than one listing
  has data, the others are reported as alternatives. prefer_indian=True
  tries .NS/.BO first (the original collector's order).
- Anything else, or a bare symbol with no data, is searched as a company
  name. Equity results keep the provider's relevance order (NSE/BSE first
  when prefer_indian=True) and the first result that has quarterly
  financial statements is used.

Errors (TickerResolutionError.code)
-----------------------------------
- empty_input        nothing was entered
- no_financial_data  the symbol exists but has no quarterly statements
- unknown_company    no listed security matched the input
- not_a_company      only ETFs/funds/indices matched (no financial statements)
Provider failures raise ProviderError (code "provider_error").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


INDIAN_SUFFIXES = (".NS", ".BO")
TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-&=^]{0,19}$")


class TickerResolutionError(ValueError):
    def __init__(self, code, message, candidates=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.candidates = candidates or []


@dataclass
class Resolution:
    input: str
    symbol: str
    method: str                      # exact | suffix | as_typed | name_search
    name: str | None = None
    exchange: str | None = None
    alternatives: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "input": self.input,
            "symbol": self.symbol,
            "method": self.method,
            "name": self.name,
            "exchange": self.exchange,
            "alternatives": self.alternatives,
            "notes": self.notes,
        }


def looks_like_ticker(text):
    return bool(TICKER_PATTERN.match(text.strip().upper()))


def is_exchange_qualified(symbol):
    return any(mark in symbol for mark in (".", "=", "^"))


def _rank_candidates(quotes, prefer_indian):
    seen = set()
    candidates = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol or symbol in seen or quote.get("quoteType") != "EQUITY":
            continue
        seen.add(symbol)
        candidates.append({
            "symbol": symbol,
            "name": quote.get("longname") or quote.get("shortname"),
            "exchange": quote.get("exchDisp") or quote.get("exchange"),
        })
    if prefer_indian:
        # Stable sort keeps the provider's relevance order within each group.
        candidates.sort(
            key=lambda c: 0 if c["symbol"].upper().endswith(INDIAN_SUFFIXES) else 1
        )
    return candidates


def resolve(user_input, provider, prefer_indian=False, max_checks=5):
    """Resolve a ticker or company name to a symbol with quarterly data."""
    text = " ".join(str(user_input or "").split())
    if not text:
        raise TickerResolutionError(
            "empty_input", "Enter a ticker (e.g. RELIANCE.NS, AAPL) or a "
            "company name."
        )

    upper = text.upper()

    if looks_like_ticker(upper) and is_exchange_qualified(upper):
        if provider.has_quarterly_financials(upper):
            return Resolution(text, upper, "exact")
        # Confirm the provider is reachable (search raises ProviderError
        # otherwise) and offer similar listings.
        candidates = _rank_candidates(provider.search(upper), prefer_indian)
        raise TickerResolutionError(
            "no_financial_data",
            f"No quarterly financial statements are available for '{upper}'. "
            "Check the exchange suffix (.NS for NSE, .BO for BSE) or try the "
            "company name.",
            candidates[:8],
        )

    if looks_like_ticker(upper):
        order = (
            [f"{upper}.NS", f"{upper}.BO", upper]
            if prefer_indian else [upper, f"{upper}.NS", f"{upper}.BO"]
        )
        found = [s for s in order if provider.has_quarterly_financials(s)]
        if found:
            symbol = found[0]
            method = "as_typed" if symbol == upper else "suffix"
            resolution = Resolution(
                text, symbol, method,
                alternatives=[{"symbol": s} for s in found[1:]],
            )
            if method == "suffix":
                resolution.notes.append(
                    f"'{upper}' was matched to {symbol}"
                    + (" (NSE)" if symbol.endswith(".NS") else " (BSE)")
                    + ". Type the full ticker to choose another listing."
                )
            if found[1:]:
                resolution.notes.append(
                    "Other listings with data: " + ", ".join(found[1:]) + "."
                )
            return resolution
        # Fall through: a single word may be a company name ("Infosys").

    quotes = provider.search(text)
    candidates = _rank_candidates(quotes, prefer_indian)
    if not candidates:
        others = [
            {"symbol": q.get("symbol"),
             "name": q.get("longname") or q.get("shortname"),
             "exchange": q.get("exchDisp") or q.get("exchange"),
             "type": q.get("quoteType")}
            for q in quotes if q.get("symbol")
        ]
        if others:
            kinds = sorted({o["type"] or "unknown" for o in others})
            raise TickerResolutionError(
                "not_a_company",
                f"'{text}' matched only non-company securities "
                f"({', '.join(kinds)}). Financial statements are available "
                "for company shares only.",
                others[:8],
            )
        raise TickerResolutionError(
            "unknown_company",
            f"No listed company matched '{text}'. Try the exact ticker, "
            "e.g. RELIANCE.NS, TCS.NS or AAPL.",
        )

    for candidate in candidates[:max_checks]:
        if provider.has_quarterly_financials(candidate["symbol"]):
            others = [c for c in candidates[:8] if c is not candidate]
            resolution = Resolution(
                text, candidate["symbol"], "name_search",
                name=candidate["name"], exchange=candidate["exchange"],
                alternatives=others,
            )
            resolution.notes.append(
                f"'{text}' was matched by name search to "
                f"{candidate['symbol']} ({candidate['name']}, "
                f"{candidate['exchange']}). Check this is the intended "
                "company; pass the exact ticker to choose another listing."
            )
            return resolution

    raise TickerResolutionError(
        "no_financial_data",
        f"Companies matching '{text}' were found, but none of the top "
        f"{min(max_checks, len(candidates))} listings has quarterly financial "
        "statements from the provider.",
        candidates[:8],
    )
