"""Annual financial panel for all US SEC filers, from XBRL "frames".

One frames request returns one value per company for a concept and period
(about 6,000 companies), so 16 years of data for every filer needs only a
few hundred requests instead of one large download per company.

  https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/CY2019.json    annual flows
  https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/CY2019Q4I.json instants

Rules
-----
- Annual flows (revenue, income, cash flow) come from the CY<year> frame.
  Its `end` is the company's fiscal year end.
- Balance-sheet values are taken from the instant frame of the calendar
  quarter containing that fiscal year end, matched on the exact date
  (within 7 days), so May or September year ends line up correctly.
- Frames hold the most recently filed value for each period, so a few
  values may be restated figures (not as first reported). Filing dates are
  not included; availability is approximated as fiscal year end + 90 days
  (the latest 10-K deadline).

Output: data/raw/sec_edgar/annual_panel.csv, one row per company and
fiscal year, with canonical column names (see CANONICAL).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_collection.sec_edgar import EdgarClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANEL_FILE = PROJECT_ROOT / "data" / "raw" / "sec_edgar" / "annual_panel.csv"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json"

FIRST_YEAR, LAST_YEAR = 2010, 2025

# canonical name -> XBRL concepts (first available wins, except revenue,
# where the largest value is used: concepts filed together, total vs parts)
FLOW_CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "gross_profit": ["GrossProfit"],
}
INSTANT_CONCEPTS = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
}
CANONICAL = list(FLOW_CONCEPTS) + list(INSTANT_CONCEPTS)


def _frame(client, concept, period):
    url = FRAMES_URL.format(concept=concept, period=period)
    data = client._get_json(url, f"frames/{concept}_{period}.json")
    if not data:
        return pd.DataFrame(columns=["cik", "end", "val", "entityName"])
    df = pd.DataFrame(data["data"])
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df


def prefetch(client, first_year, last_year, workers=8, echo=print):
    """Download every needed frame into the cache, in parallel (the client
    enforces the SEC rate limit across threads)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = [(c, f"CY{y}") for concepts in FLOW_CONCEPTS.values() for c in concepts
            for y in range(first_year, last_year + 1)]
    jobs += [(c, f"CY{y}Q{q}I") for concepts in INSTANT_CONCEPTS.values() for c in concepts
             for y in range(first_year, last_year + 2) for q in (1, 2, 3, 4)]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(client._get_json,
                               FRAMES_URL.format(concept=c, period=p),
                               f"frames/{c}_{p}.json") for c, p in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                echo(f"  downloaded {done}/{len(jobs)} frames")


def download_panel(client=None, first_year=FIRST_YEAR, last_year=LAST_YEAR,
                   echo=print):
    """Download all frames and assemble the annual panel."""
    client = client or EdgarClient(max_cache_age_days=365)
    prefetch(client, first_year, last_year, echo=echo)
    years = range(first_year, last_year + 1)

    # Annual flows -------------------------------------------------------
    rows = {}
    names = {}
    for name, concepts in FLOW_CONCEPTS.items():
        echo(f"  flows: {name}")
        for year in years:
            for concept in concepts:
                df = _frame(client, concept, f"CY{year}")
                for r in df.itertuples():
                    key = (int(r.cik), r.end)
                    names[int(r.cik)] = r.entityName
                    entry = rows.setdefault(key, {"cik": int(r.cik), "fiscal_year_end": r.end})
                    current = entry.get(name)
                    if name == "revenue":
                        if current is None or r.val > current:
                            entry[name] = float(r.val)
                    elif current is None:
                        entry[name] = float(r.val)

    panel = pd.DataFrame(rows.values())
    panel["company"] = panel["cik"].map(names)

    # Instants matched to the fiscal year end ------------------------------
    quarters = [f"CY{y}Q{q}I" for y in range(first_year, last_year + 2) for q in (1, 2, 3, 4)]
    for name, concepts in INSTANT_CONCEPTS.items():
        echo(f"  instants: {name}")
        values = {}
        for concept in reversed(concepts):        # earlier concepts win
            for period in quarters:
                df = _frame(client, concept, period)
                for r in df.itertuples():
                    values[(int(r.cik), r.end)] = float(r.val)
        lookup = values  # plain dict: fast (cik, date) lookups
        matched = []
        for r in panel[["cik", "fiscal_year_end"]].itertuples(index=False):
            value = lookup.get((r.cik, r.fiscal_year_end))
            if value is None:
                for shift in (1, -1, 2, -2, 3, -3, 7, -7):
                    value = lookup.get((r.cik, r.fiscal_year_end + pd.Timedelta(days=shift)))
                    if value is not None:
                        break
            matched.append(np.nan if value is None else value)
        panel[name] = matched

    panel = panel.sort_values(["cik", "fiscal_year_end"]).reset_index(drop=True)
    panel["source"] = "SEC EDGAR frames"
    return panel[["cik", "company", "fiscal_year_end"] + CANONICAL + ["source"]]


def load_or_download(path=PANEL_FILE, refresh=False, echo=print):
    path = Path(path)
    if path.exists() and not refresh:
        panel = pd.read_csv(path, parse_dates=["fiscal_year_end"])
        echo(f"Loaded annual panel: {len(panel)} rows, {panel['cik'].nunique()} companies")
        return panel
    echo("Downloading SEC EDGAR frames (several hundred requests, cached)...")
    panel = download_panel(echo=echo)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
    echo(f"Saved {path} ({len(panel)} rows, {panel['cik'].nunique()} companies)")
    return panel
