"""Market Agent: share-price risk and valuation context.

From about two years of daily prices (Yahoo Finance) for the stock and its
benchmark index (S&P 500 for US listings, Nifty 50 for NSE/BSE):
  - annualised volatility (1 year), maximum drawdown (1 year)
  - beta and correlation to the index (1 year of daily returns)
  - 3, 6 and 12-month returns, and 12-month return relative to the index
  - distance from the 52-week high
plus valuation from the company profile (market cap, P/E, P/B).

Signals use fixed thresholds stated in each signal's detail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.common import num, result, signal
from src.data_collection.provider import ProviderError


THRESHOLDS = {
    "volatility_high": 0.50, "volatility_elevated": 0.35,
    "drawdown_high": 0.40, "drawdown_elevated": 0.25,
    "beta_high": 1.5, "relative_return_weak": -0.20,
    "relative_return_strong": 0.20, "near_low": 0.15,
}


def benchmark_for(symbol):
    return "^NSEI" if symbol.upper().endswith((".NS", ".BO")) else "^GSPC"


def _returns(close):
    return close.pct_change().dropna()


def _period_return(close, days):
    if len(close) < 2:
        return None
    start = close.index[-1] - pd.Timedelta(days=days)
    window = close[close.index >= start]
    if len(window) < 2:
        return None
    return float(window.iloc[-1] / window.iloc[0] - 1)


def compute_metrics(prices, benchmark=None):
    close = prices["Close"].dropna()
    if len(close) < 60:
        return None
    last_year = close[close.index >= close.index[-1] - pd.Timedelta(days=365)]
    r = _returns(last_year)
    running_max = last_year.cummax()
    drawdown = float((last_year / running_max - 1).min())
    high_52w = float(last_year.max())
    low_52w = float(last_year.min())
    metrics = {
        "last_close": num(close.iloc[-1], 4),
        "as_of": str(close.index[-1].date()),
        "volatility_1y": num(r.std() * np.sqrt(252), 4),
        "max_drawdown_1y": num(drawdown, 4),
        "return_3m": num(_period_return(close, 91), 4),
        "return_6m": num(_period_return(close, 182), 4),
        "return_12m": num(_period_return(close, 365), 4),
        "below_52w_high": num(1 - close.iloc[-1] / high_52w, 4),
        "above_52w_low": num(close.iloc[-1] / low_52w - 1, 4),
        "beta_1y": None, "correlation_1y": None, "benchmark_return_12m": None,
        "relative_return_12m": None,
    }
    if benchmark is not None and not benchmark.empty:
        b_close = benchmark["Close"].dropna()
        joined = pd.concat([_returns(close), _returns(b_close)], axis=1, join="inner").dropna()
        joined = joined[joined.index >= close.index[-1] - pd.Timedelta(days=365)]
        if len(joined) > 60:
            stock, index = joined.iloc[:, 0], joined.iloc[:, 1]
            metrics["beta_1y"] = num(stock.cov(index) / index.var(), 3)
            metrics["correlation_1y"] = num(stock.corr(index), 3)
        bench_12m = _period_return(b_close, 365)
        metrics["benchmark_return_12m"] = num(bench_12m, 4)
        if bench_12m is not None and metrics["return_12m"] is not None:
            metrics["relative_return_12m"] = num(metrics["return_12m"] - bench_12m, 4)
    return metrics


def market_signals(m, profile):
    t, out = THRESHOLDS, []
    vol, dd, beta = m.get("volatility_1y"), m.get("max_drawdown_1y"), m.get("beta_1y")
    if vol is not None:
        if vol >= t["volatility_high"]:
            out.append(signal("high_volatility", "high", "negative", "Very volatile share price",
                              f"Annualised volatility {vol:.0%} (threshold {t['volatility_high']:.0%})."))
        elif vol >= t["volatility_elevated"]:
            out.append(signal("elevated_volatility", "medium", "negative", "Elevated volatility",
                              f"Annualised volatility {vol:.0%} (threshold {t['volatility_elevated']:.0%})."))
    if dd is not None:
        if -dd >= t["drawdown_high"]:
            out.append(signal("deep_drawdown", "high", "negative", "Deep price fall in the last year",
                              f"Largest peak-to-trough fall {dd:.0%} (threshold -{t['drawdown_high']:.0%})."))
        elif -dd >= t["drawdown_elevated"]:
            out.append(signal("drawdown", "medium", "negative", "Significant price fall in the last year",
                              f"Largest peak-to-trough fall {dd:.0%} (threshold -{t['drawdown_elevated']:.0%})."))
    if beta is not None and beta >= t["beta_high"]:
        out.append(signal("high_beta", "medium", "negative", "Moves much more than the market",
                          f"Beta {beta:.2f} to the index (threshold {t['beta_high']})."))
    rel = m.get("relative_return_12m")
    if rel is not None:
        if rel <= t["relative_return_weak"]:
            out.append(signal("underperformance", "medium", "negative", "Lagging the market",
                              f"12-month return {rel:+.0%} relative to the index."))
        elif rel >= t["relative_return_strong"]:
            out.append(signal("outperformance", "info", "positive", "Ahead of the market",
                              f"12-month return {rel:+.0%} relative to the index."))
    low = m.get("above_52w_low")
    if low is not None and low <= t["near_low"]:
        out.append(signal("near_52w_low", "low", "negative", "Trading near its 52-week low",
                          f"{low:.0%} above the 52-week low."))
    pe = num(profile.get("trailingPE"))
    if pe is not None and pe > 60:
        out.append(signal("high_valuation", "low", "neutral", "High earnings multiple",
                          f"Trailing P/E {pe:.0f}; expectations are high."))
    return out


def run(symbol, provider, profile=None):
    profile = profile or {}
    try:
        prices = provider.price_history(symbol)
        bench_symbol = benchmark_for(symbol)
        try:
            bench = provider.price_history(bench_symbol)
        except ProviderError:
            bench = pd.DataFrame()
    except ProviderError as exc:
        return result("market", "error", f"Price data could not be downloaded: {exc}",
                      errors=[str(exc)])
    if prices is None or prices.empty:
        return result("market", "unavailable", "No price history is available for this symbol.")
    m = compute_metrics(prices, bench)
    if m is None:
        return result("market", "unavailable", "Not enough price history (under 60 trading days).")
    m["benchmark"] = bench_symbol
    m["valuation"] = {k: num(profile.get(k), 4) for k in
                      ("marketCap", "trailingPE", "forwardPE", "priceToBook", "dividendYield")}
    signals = market_signals(m, profile)
    parts = []
    if m["return_12m"] is not None:
        parts.append(f"The share price moved {m['return_12m']:+.0%} over 12 months"
                     + (f" ({m['relative_return_12m']:+.0%} vs {bench_symbol})"
                        if m["relative_return_12m"] is not None else "") + ".")
    if m["volatility_1y"] is not None:
        parts.append(f"Annualised volatility is {m['volatility_1y']:.0%} and the largest fall "
                     f"in the last year was {m['max_drawdown_1y']:.0%}.")
    if m["beta_1y"] is not None:
        parts.append(f"Beta to the index is {m['beta_1y']:.2f}.")
    status = "ok" if m["beta_1y"] is not None else "partial"
    return result("market", status, " ".join(parts), signals, m,
                  sources=[f"Yahoo Finance daily prices for {symbol} and {bench_symbol}"])
