from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


# Resolve every file path from the project root, not from the terminal folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "multi_company_financial_dataset.csv"

# Keep the original starter companies so your existing development dataset remains usable.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM"
]

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


def get_metric(statement: pd.DataFrame, metric: str) -> pd.Series:
    """Return a numeric statement row, or an empty/missing row if unavailable."""
    if statement is None or statement.empty or metric not in statement.index:
        if statement is None:
            return pd.Series(dtype="float64")
        return pd.Series(float("nan"), index=statement.columns, dtype="float64")

    values = statement.loc[metric]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[0]
    return pd.to_numeric(values, errors="coerce")


def _has_financial_statements(symbol: str) -> bool:
    """Check whether Yahoo Finance exposes an income statement for this symbol."""
    try:
        return not yf.Ticker(symbol).income_stmt.empty
    except Exception:
        return False


def _search_company_name(query: str) -> Optional[str]:
    """Search Yahoo Finance for a company name and let the user choose a result."""
    try:
        response = requests.get(
            YAHOO_SEARCH_URL,
            params={"q": query, "quotesCount": 10, "newsCount": 0},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except Exception as exc:
        print(f"Company-name search could not be completed: {exc}")
        print("Please enter an exchange ticker instead, e.g. RELIANCE.NS or AAPL.")
        return None

    # Prefer company shares, and put Indian NSE/BSE symbols near the top.
    candidates = []
    seen = set()
    for quote in quotes:
        symbol = quote.get("symbol")
        quote_type = quote.get("quoteType", "")
        if not symbol or quote_type not in {"EQUITY", "ETF"} or symbol in seen:
            continue
        seen.add(symbol)
        candidates.append(quote)

    candidates.sort(
        key=lambda item: 0 if item.get("symbol", "").endswith((".NS", ".BO")) else 1
    )

    if not candidates:
        print("No matching listed company was found. Try its exact ticker symbol.")
        return None

    print("\nMatching securities:")
    for index, item in enumerate(candidates[:8], start=1):
        name = item.get("shortname") or item.get("longname") or "Name unavailable"
        exchange = item.get("exchDisp") or item.get("exchange") or "Exchange unknown"
        print(f"{index}. {name} | {item['symbol']} | {exchange}")

    while True:
        choice = input("Choose the matching number (or press Enter to cancel): ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= min(8, len(candidates)):
            return candidates[int(choice) - 1]["symbol"]
        print("Please enter a number shown in the list, or press Enter to cancel.")


def resolve_ticker(user_input: str) -> Optional[str]:
    """Resolve a ticker or company name, preferring NSE/BSE for bare Indian symbols."""
    value = user_input.strip().upper()
    if not value:
        return None

    # An explicit exchange-qualified ticker should be respected exactly.
    if "." in value or "=" in value or "^" in value:
        if _has_financial_statements(value):
            return value
        print(f"No income-statement data was found for ticker '{value}'.")
        return None

    # A bare symbol could be an NSE or BSE symbol, or an international symbol.
    for candidate in (f"{value}.NS", f"{value}.BO", value):
        if _has_financial_statements(candidate):
            return candidate

    # If the input looks like a company name, search it and ask the user to confirm.
    if " " in user_input.strip() or not value.isalnum():
        selected = _search_company_name(user_input.strip())
        if selected and _has_financial_statements(selected):
            return selected
        if selected:
            print(f"Yahoo Finance did not provide an income statement for '{selected}'.")
        return None

    print(f"Could not find financial statements for '{value}'.")
    print("Try a full ticker such as RELIANCE.NS, TCS.NS, HDFCBANK.NS, or AAPL.")
    print("Not every listed company has financial statements available through Yahoo Finance.")
    return None


def collect_company_data(ticker: str) -> Optional[pd.DataFrame]:
    """Collect annual financial statements and calculate basic financial indicators."""
    print(f"\nCollecting data for: {ticker}")
    stock = yf.Ticker(ticker)

    income_data = stock.income_stmt
    balance_data = stock.balance_sheet
    cashflow_data = stock.cashflow

    if income_data is None or income_data.empty:
        print(f"No income statement data found for {ticker}; skipping it.")
        return None

    data = pd.DataFrame({
        "Total Revenue": get_metric(income_data, "Total Revenue"),
        "Gross Profit": get_metric(income_data, "Gross Profit"),
        "Operating Income": get_metric(income_data, "Operating Income"),
        "Net Income": get_metric(income_data, "Net Income"),
        "EBITDA": get_metric(income_data, "EBITDA"),
        "Operating Expense": get_metric(income_data, "Operating Expense"),
        "Cost Of Revenue": get_metric(income_data, "Cost Of Revenue"),
        "Research And Development": get_metric(income_data, "Research And Development"),
        "Interest Expense": get_metric(income_data, "Interest Expense"),
        "Total Debt": get_metric(balance_data, "Total Debt"),
        "Cash And Cash Equivalents": get_metric(balance_data, "Cash And Cash Equivalents"),
        "Tangible Book Value": get_metric(balance_data, "Tangible Book Value"),
        "Free Cash Flow": get_metric(cashflow_data, "Free Cash Flow"),
        "Operating Cash Flow": get_metric(cashflow_data, "Operating Cash Flow"),
        "Capital Expenditure": get_metric(cashflow_data, "Capital Expenditure"),
    })

    # Important: sort reporting years BEFORE calculating growth. Yahoo statements
    # are often newest-first; calculating pct_change before sorting reverses growth.
    data.index = pd.to_datetime(data.index, errors="coerce")
    data.index.name = "Date"
    data = data.loc[~data.index.isna()].sort_index().copy()

    if data.empty:
        print(f"No usable dated financial rows found for {ticker}; skipping it.")
        return None

    def ratio(numerator: str, denominator: str) -> pd.Series:
        denominator_values = data[denominator].where(data[denominator] != 0)
        return data[numerator] / denominator_values

    data["Profit Margin"] = ratio("Net Income", "Total Revenue")
    data["Operating Margin"] = ratio("Operating Income", "Total Revenue")
    data["EBITDA Margin"] = ratio("EBITDA", "Total Revenue")
    data["Gross Profit Margin"] = ratio("Gross Profit", "Total Revenue")
    data["Debt to Cash"] = ratio("Total Debt", "Cash And Cash Equivalents")
    data["Free Cash Flow Margin"] = ratio("Free Cash Flow", "Total Revenue")
    data["Operating Cash Flow Margin"] = ratio("Operating Cash Flow", "Total Revenue")

    data["Revenue Growth"] = data["Total Revenue"].pct_change(fill_method=None)
    data["Net Income Growth"] = data["Net Income"].pct_change(fill_method=None)
    data["Debt Growth"] = data["Total Debt"].pct_change(fill_method=None)
    data["Cash Growth"] = data["Cash And Cash Equivalents"].pct_change(fill_method=None)

    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data["Ticker"] = ticker
    data = data.reset_index()
    return data


def save_or_merge_dataset(new_data: pd.DataFrame) -> None:
    """Merge new company rows into the existing dataset without deleting old tickers."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            combined = pd.concat([existing, new_data], ignore_index=True, sort=False)
        except Exception as exc:
            print(f"Could not read the existing dataset ({exc}). Saving only new rows.")
            combined = new_data.copy()
    else:
        combined = new_data.copy()

    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Ticker", "Date"])
    combined = combined.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset saved successfully:")
    print(f"File: {OUTPUT_FILE}")
    print(f"Shape: {combined.shape}")
    print("Tickers in dataset:", ", ".join(sorted(combined["Ticker"].astype(str).unique())))


def main() -> None:
    print("=" * 68)
    print("DYNAMIC MULTI-COMPANY FINANCIAL DATA COLLECTION")
    print("Supports available international and Indian listed-company tickers")
    print("=" * 68)
    print("Examples: RELIANCE, TCS, HDFCBANK, INFY, AAPL, TSLA")
    print("You may also enter a company name, then choose a search result.")
    print("Enter multiple names/tickers separated by commas.")
    print("Press Enter without typing anything to collect the original starter tickers.")

    raw_input = input("Company name(s) or ticker(s): ").strip()
    requested = [item.strip() for item in raw_input.split(",") if item.strip()]
    if not requested:
        requested = DEFAULT_TICKERS.copy()

    resolved_tickers = []
    for item in requested:
        ticker = resolve_ticker(item)
        if ticker and ticker not in resolved_tickers:
            resolved_tickers.append(ticker)

    if not resolved_tickers:
        print("No valid tickers were resolved. No files were changed.")
        return

    collected = []
    for ticker in resolved_tickers:
        try:
            company_data = collect_company_data(ticker)
            if company_data is not None and not company_data.empty:
                collected.append(company_data)
                print(f"Collected {len(company_data)} annual rows for {ticker}.")
        except Exception as exc:
            print(f"Error collecting {ticker}: {exc}")
        time.sleep(1)

    if not collected:
        print("No new financial data was collected. The existing dataset was not changed.")
        return

    save_or_merge_dataset(pd.concat(collected, ignore_index=True))
    print("\nCollection finished. Note: availability varies by company and data provider.")


if __name__ == "__main__":
    main()
