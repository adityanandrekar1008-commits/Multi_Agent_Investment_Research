"""Quarterly history from SEC EDGAR XBRL facts, as originally reported.

For each US company this builds one row per fiscal quarter with:
  Date (period end), Filed (date the figures became public), Total Revenue,
  Net Income, Operating Income, Total Assets, Total Debt, Stockholders
  Equity, Current Assets, Current Liabilities, and the XBRL concept used
  for revenue.

Rules
-----
- Only 10-Q / 10-K filings (and amendments) are used.
- Point in time: when the same period appears in several filings (later
  filings repeat prior-year comparatives, sometimes restated), the FIRST
  filed value is kept, together with its filing date. This avoids training
  on restated numbers that were not known at the time.
- Quarterly flows (revenue, income) come from 3-month facts (80-100 days,
  covering 13- and 14-week quarters). Fourth quarters are taken as the
  annual value minus the 9-month year-to-date value from the same fiscal
  year (or minus the three reported quarters), even when the 10-K also tags
  a 3-month Q4 figure, because such figures are sometimes segment values
  tagged as totals. Q4's filing date is the 10-K's.
- Revenue: several XBRL concepts are used across companies and years
  (e.g. SalesRevenueNet until 2018, then RevenueFromContractWithCustomer...).
  For each quarter the earliest-filed value across concepts is used; among
  concepts filed on the same day, the largest (total rather than goods-only).
  Taking the largest overall would prefer later restatements.
- Point_In_Time is False when revenue was first filed more than
  MAX_FILING_LAG_DAYS after the quarter ended. Such rows (mostly before a
  company's first XBRL filing) mix restated and original figures and are
  excluded from modelling.
- Total Debt (approximate): LongTermDebt (or noncurrent + current portions)
  plus CommercialPaper (or ShortTermBorrowings). Leases are excluded, so it
  is lower than Yahoo Finance's "Total Debt", which includes them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sec_edgar" / "quarterly"

# The 7 US companies in the project (JPM is a bank and is excluded from the
# model; RELIANCE.NS is not an SEC filer).
DEFAULT_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]
FLOW_CONCEPTS = {
    "Net Income": ["NetIncomeLoss", "ProfitLoss"],
    "Operating Income": ["OperatingIncomeLoss"],
}
INSTANT_CONCEPTS = {
    "Total Assets": ["Assets"],
    "Stockholders Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Current Assets": ["AssetsCurrent"],
    "Current Liabilities": ["LiabilitiesCurrent"],
}
QUARTER_DAYS = (80, 100)
YEAR_DAYS = (350, 380)
NINE_MONTH_DAYS = (260, 285)

# A 10-Q is due 40-45 days and a 10-K 60-90 days after period end. A
# quarter whose revenue was first filed later than this only appears as a
# comparative in a later filing (e.g. before the company started XBRL
# tagging in 2009-2011), possibly restated, so it is not point-in-time.
MAX_FILING_LAG_DAYS = 120


def _facts(us_gaap, concept):
    """All USD facts for a concept from 10-Q/10-K filings, as a DataFrame."""
    units = us_gaap.get(concept, {}).get("units", {})
    rows = units.get("USD", [])
    if not rows:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "form"])
    df = pd.DataFrame(rows)
    df = df[df["form"].astype(str).str.match(r"^10-[QK]")]
    for column in ("start", "end", "filed"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    if "start" not in df.columns:
        df["start"] = pd.NaT
    return df.dropna(subset=["end", "val", "filed"])


def _first_filed(df, keys):
    return (df.sort_values("filed")
              .drop_duplicates(subset=keys, keep="first")
              .reset_index(drop=True))


def quarterly_flow(us_gaap, concept):
    """{period_end: (value, filed)} for a flow concept, with derived Q4s."""
    df = _facts(us_gaap, concept)
    df = df.dropna(subset=["start"])
    if df.empty:
        return {}
    df = _first_filed(df, ["start", "end"])
    df["days"] = (df["end"] - df["start"]).dt.days

    quarters = {
        row.end: (float(row.val), row.filed)
        for row in df[df["days"].between(*QUARTER_DAYS)].itertuples()
    }
    nine_months = df[df["days"].between(*NINE_MONTH_DAYS)]
    for year in df[df["days"].between(*YEAR_DAYS)].itertuples():
        ytd = nine_months[nine_months["start"] == year.start]
        if not ytd.empty:
            # Prefer annual minus 9-month YTD even when a 3-month Q4 fact
            # exists: Q4 first appears in the 10-K, and some 10-Ks tag
            # segment-level quarterly figures without a dimension (e.g.
            # Amazon's 2012 10-K reports 17.4B for Q4 2012; the derived
            # 61.09B - 39.83B = 21.27B matches its later filings).
            value = year.val - ytd.iloc[0]["val"]
        elif year.end in quarters:
            continue
        else:
            inside = [
                (end, v) for end, v in quarters.items()
                if year.start < end < year.end
            ]
            if len(inside) != 3:
                continue
            value = year.val - sum(v[0] for _, v in inside)
        quarters[year.end] = (float(value), year.filed)
    return quarters


def instant(us_gaap, concept):
    """{date: (value, filed)} for a balance-sheet concept."""
    df = _facts(us_gaap, concept)
    if df.empty:
        return {}
    df = _first_filed(df, ["end"])
    return {row.end: (float(row.val), row.filed) for row in df.itertuples()}


def _first_available(us_gaap, concepts, reader):
    """Merge concepts in priority order (earlier concepts win per period)."""
    merged = {}
    for concept in reversed(concepts):
        merged.update(reader(us_gaap, concept))
    return merged


def _total_debt(us_gaap):
    long_term = instant(us_gaap, "LongTermDebt")
    noncurrent = instant(us_gaap, "LongTermDebtNoncurrent")
    current = _first_available(
        us_gaap, ["LongTermDebtCurrent", "DebtCurrent"], instant)
    short_term = _first_available(
        us_gaap, ["CommercialPaper", "ShortTermBorrowings"], instant)
    dates = set(long_term) | set(noncurrent)
    debt = {}
    for date in dates:
        if date in long_term:
            value, filed = long_term[date]
        else:
            value, filed = noncurrent[date]
            if date in current:
                value += current[date][0]
        if date in short_term:
            value += short_term[date][0]
        debt[date] = (value, filed)
    return debt


def company_quarters(ticker, facts):
    """One row per fiscal quarter for a company-facts JSON document."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    # Per quarter: the earliest-filed revenue across concepts (point in
    # time); among concepts filed together, the largest (a total rather than
    # e.g. goods-only revenue).
    revenue = {}
    for concept in REVENUE_CONCEPTS:
        for end, (value, filed) in quarterly_flow(us_gaap, concept).items():
            best = revenue.get(end)
            if (best is None or filed < best[1]
                    or (filed == best[1] and value > best[0])):
                revenue[end] = (value, filed, concept)
    if not revenue:
        return pd.DataFrame()

    columns = {
        name: _first_available(us_gaap, concepts, quarterly_flow)
        for name, concepts in FLOW_CONCEPTS.items()
    }
    columns.update({
        name: _first_available(us_gaap, concepts, instant)
        for name, concepts in INSTANT_CONCEPTS.items()
    })
    columns["Total Debt"] = _total_debt(us_gaap)

    rows = []
    for end in sorted(revenue):
        value, filed, concept = revenue[end]
        lag = (filed - end).days
        row = {
            "Date": end.date(), "Ticker": ticker, "Filed": filed.date(),
            "Days_To_Filing": lag,
            "Point_In_Time": lag <= MAX_FILING_LAG_DAYS,
            "Total Revenue": value, "Revenue_Concept": concept,
        }
        for name, series in columns.items():
            row[name] = series[end][0] if end in series else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table["Source"] = "SEC EDGAR"
    return table


CROSS_CHECK_FIELDS = [
    "Total Revenue", "Net Income", "Operating Income", "Total Assets",
    "Stockholders Equity", "Current Assets", "Current Liabilities", "Total Debt",
]


def cross_check(tables, yahoo_combined_file, tolerance=0.005, max_days=7):
    """Compare EDGAR quarters with Yahoo Finance quarters for the same
    company (period ends within `max_days`). Both come from the same filings,
    so most fields should agree within `tolerance`; Total Debt is defined
    differently (Yahoo includes leases) and is reported for information."""
    yahoo = pd.read_csv(yahoo_combined_file, parse_dates=["Date"])
    stats = {field: {"compared": 0, "matching": 0} for field in CROSS_CHECK_FIELDS}
    mismatches = []
    for ticker, table in tables.items():
        edgar = table.copy()
        edgar["Date"] = pd.to_datetime(edgar["Date"])
        for _, y in yahoo[yahoo["Ticker"] == ticker].iterrows():
            match = edgar[(edgar["Date"] - y["Date"]).abs().dt.days <= max_days]
            if match.empty:
                continue
            e = match.iloc[0]
            for field in CROSS_CHECK_FIELDS:
                if field not in yahoo.columns or field not in edgar.columns:
                    continue
                if pd.isna(y[field]) or pd.isna(e[field]):
                    continue
                stats[field]["compared"] += 1
                same = abs(e[field] - y[field]) <= tolerance * abs(y[field])
                stats[field]["matching"] += int(same)
                if not same and field != "Total Debt":
                    mismatches.append({
                        "ticker": ticker, "period": str(y["Date"].date()),
                        "field": field, "yahoo": float(y[field]),
                        "edgar": float(e[field]),
                    })
    return {"tolerance": tolerance, "fields": stats, "mismatches": mismatches}


def collect(client, tickers=None, output_dir=OUTPUT_DIR, echo=print):
    """Download and save quarterly history per company. Returns
    ({ticker: DataFrame}, {ticker: note})."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, notes = {}, {}
    for ticker in tickers or DEFAULT_TICKERS:
        cik, facts = client.company_facts(ticker)
        if facts is None:
            notes[ticker] = "not an SEC filer (no CIK); skipped"
            echo(f"  {ticker}: {notes[ticker]}")
            continue
        table = company_quarters(ticker, facts)
        if table.empty:
            notes[ticker] = "no quarterly revenue facts found"
            echo(f"  {ticker}: {notes[ticker]}")
            continue
        table.to_csv(output_dir / f"{ticker}_quarterly.csv", index=False)
        tables[ticker] = table
        notes[ticker] = (f"CIK {cik}: {len(table)} quarters, "
                         f"{table['Date'].min()} to {table['Date'].max()}")
        echo(f"  {ticker}: {notes[ticker]}")
    return tables, notes
