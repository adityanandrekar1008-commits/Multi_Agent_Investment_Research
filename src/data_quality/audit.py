"""Data-quality audit for quarterly company financial data.

The audit only reads data. It never changes, fills, or removes values.

Checks per company:
- duplicate company/date records
- empty periods (provider returned a date with no income-statement values)
- missing values in the fields the model and report depend on
- quarterly gaps (missing quarters between reports)
- stale data (latest period is older than one quarter plus a filing window)
- short history
- fiscal calendar (quarters that do not end in Mar/Jun/Sep/Dec)
- outliers and one-off events (large revenue jumps, extreme margins,
  negative equity, debt above assets, unusual items in net income)
- accounting identities (assets = liabilities + equity, working capital,
  free cash flow = operating cash flow + capital expenditure)

Cross-company checks:
- mixed reporting currencies
- mixed fiscal calendars
- companies whose statements look like a bank/financial company
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


QUARTER_DAYS = (70, 110)

# One quarter (~91 days) plus a ~45-day filing window.
STALE_AFTER_DAYS = 135

MIN_QUARTERS_FOR_YOY = 5
MIN_QUARTERS_RECOMMENDED = 8

INCOME_FIELDS = ["Total Revenue", "Net Income", "Operating Income"]
BALANCE_FIELDS = [
    "Total Assets",
    "Total Debt",
    "Stockholders Equity",
    "Current Assets",
    "Current Liabilities",
]
CASH_FLOW_FIELDS = ["Operating Cash Flow", "Free Cash Flow"]
KEY_FIELDS = INCOME_FIELDS + BALANCE_FIELDS + CASH_FLOW_FIELDS

REVENUE_JUMP_LIMIT = 0.50        # |quarter-on-quarter change| above 50%
UNUSUAL_ITEMS_LIMIT = 0.10       # unusual items above 10% of net income
IDENTITY_TOLERANCE = 0.05        # 5% of total assets
SMALL_TOLERANCE = 0.01           # 1% for working capital / free cash flow

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def _finding(severity, check, ticker, message, period=None, value=None):
    return {
        "severity": severity,
        "check": check,
        "ticker": ticker,
        "period": None if period is None else str(pd.Timestamp(period).date()),
        "message": message,
        "value": None if value is None or pd.isna(value) else float(value),
    }


def infer_currency(ticker):
    """Best-effort currency guess from the exchange suffix (used only when the
    provider's reported currency is not available)."""
    suffix_currency = {
        ".NS": "INR", ".BO": "INR", ".L": "GBp", ".T": "JPY",
        ".HK": "HKD", ".TO": "CAD", ".DE": "EUR", ".PA": "EUR",
    }
    for suffix, currency in suffix_currency.items():
        if str(ticker).upper().endswith(suffix):
            return currency
    return "USD"


RAW_EXTRA_FIELDS = {
    # one-off events
    "Total Unusual Items": "income_statement",
    # assets = liabilities + equity including minority interest
    "Total Equity Gross Minority Interest": "balance_sheet",
}


def load_raw_extras(raw_dir, tickers):
    """Read audit-only fields that the combined dataset does not keep from
    the raw quarterly statements. Returns Date, Ticker + RAW_EXTRA_FIELDS."""
    columns = ["Date", "Ticker"] + list(RAW_EXTRA_FIELDS)
    if raw_dir is None:
        return pd.DataFrame(columns=columns)

    per_ticker = []
    for ticker in tickers:
        merged = None
        for statement in sorted(set(RAW_EXTRA_FIELDS.values())):
            path = Path(raw_dir) / f"{ticker}_{statement}.csv"
            if not path.exists():
                continue
            raw = pd.read_csv(path)
            fields = [
                f for f, s in RAW_EXTRA_FIELDS.items()
                if s == statement and f in raw.columns
            ]
            if not fields:
                continue
            part = raw[["Date"] + fields].copy()
            part["Date"] = pd.to_datetime(part["Date"], errors="coerce")
            part = part.dropna(subset=["Date"]).drop_duplicates("Date", keep="last")
            merged = part if merged is None else merged.merge(
                part, on="Date", how="outer"
            )
        if merged is not None:
            merged["Ticker"] = ticker
            per_ticker.append(merged)

    if not per_ticker:
        return pd.DataFrame(columns=columns)
    extras = pd.concat(per_ticker, ignore_index=True)
    for column in columns:
        if column not in extras.columns:
            extras[column] = np.nan
    return extras[columns]


def _audit_company(ticker, data, as_of, extras):
    findings = []
    data = data.sort_values("Date").reset_index(drop=True)
    present = [field for field in KEY_FIELDS if field in data.columns]

    # Duplicates
    duplicates = data[data.duplicated("Date", keep=False)]
    for period in duplicates["Date"].unique():
        findings.append(_finding(
            "error", "duplicate_period", ticker,
            "More than one record for the same reporting date.", period,
        ))

    # Empty periods: a date with no income-statement values at all.
    income_present = [f for f in INCOME_FIELDS if f in data.columns]
    empty_mask = data[income_present].isna().all(axis=1) if income_present \
        else pd.Series(True, index=data.index)
    latest_date = data["Date"].max()
    for period in data.loc[empty_mask, "Date"]:
        if period == latest_date:
            findings.append(_finding(
                "warning", "incomplete_latest_period", ticker,
                "The most recent period has no income-statement values "
                "(the provider had only partial data when it was collected). "
                "Re-collect the data.", period,
            ))
        else:
            findings.append(_finding(
                "info", "empty_period", ticker,
                "Provider returned this older date without income-statement "
                "values; it is dropped before modelling.", period,
            ))
    usable = data.loc[~empty_mask].copy()
    usable = usable.merge(
        extras.drop(columns=["Ticker"]), on="Date", how="left"
    ) if not extras.empty else usable

    # Missing key fields (only counted on usable periods)
    for field in KEY_FIELDS:
        if field not in data.columns:
            findings.append(_finding(
                "warning", "missing_field", ticker,
                f"'{field}' is not provided at all for this company.",
            ))
            continue
        missing = usable[usable[field].isna()]
        if not missing.empty:
            dates = ", ".join(str(d.date()) for d in missing["Date"])
            findings.append(_finding(
                "warning", "missing_values", ticker,
                f"'{field}' missing in {len(missing)} of {len(usable)} "
                f"usable periods ({dates}).",
                value=len(missing),
            ))

    # Quarterly gaps, measured over usable periods
    dates = usable["Date"].dropna().sort_values().reset_index(drop=True)
    for previous, current in zip(dates[:-1], dates[1:]):
        days = (current - previous).days
        if days > QUARTER_DAYS[1]:
            missing_quarters = max(1, round(days / 91) - 1)
            findings.append(_finding(
                "warning", "quarterly_gap", ticker,
                f"About {missing_quarters} quarter(s) missing between "
                f"{previous.date()} and {current.date()} ({days} days).",
                current, days,
            ))
        elif days < QUARTER_DAYS[0]:
            findings.append(_finding(
                "warning", "short_interval", ticker,
                f"Only {days} days between {previous.date()} and "
                f"{current.date()}; possible fiscal-year change or "
                "duplicate period.", current, days,
            ))

    # Stale latest period
    if not dates.empty:
        latest = dates.iloc[-1]
        age = (pd.Timestamp(as_of) - latest).days
        if age > STALE_AFTER_DAYS:
            findings.append(_finding(
                "warning", "stale_data", ticker,
                f"Latest period {latest.date()} is {age} days old; a newer "
                "quarter has probably been reported. Re-collect the data.",
                latest, age,
            ))

    # History length
    if len(dates) < MIN_QUARTERS_RECOMMENDED:
        note = (
            "too few for year-over-year comparison"
            if len(dates) < MIN_QUARTERS_FOR_YOY
            else "enough for one year-over-year comparison only"
        )
        findings.append(_finding(
            "info", "short_history", ticker,
            f"Only {len(dates)} usable quarters ({note}).",
            value=len(dates),
        ))

    # Fiscal calendar
    months = sorted(set(dates.dt.month))
    if months and not set(months) <= {3, 6, 9, 12}:
        findings.append(_finding(
            "info", "fiscal_calendar", ticker,
            "Fiscal quarters end in months "
            f"{months}; periods are not aligned with calendar quarters "
            "used by other companies.",
        ))

    # Outliers and unusual values
    if "Total Revenue" in usable.columns:
        revenue = usable.set_index("Date")["Total Revenue"]
        change = revenue.pct_change(fill_method=None)
        for period, value in change.items():
            if pd.notna(value) and abs(value) > REVENUE_JUMP_LIMIT:
                findings.append(_finding(
                    "warning", "revenue_jump", ticker,
                    f"Revenue changed {value:+.0%} from the previous period; "
                    "check for a one-off event or a reporting change.",
                    period, value,
                ))

    for numerator, label in [
        ("Net Income", "Net income margin"),
        ("Operating Income", "Operating margin"),
    ]:
        if numerator in usable.columns and "Total Revenue" in usable.columns:
            revenue = usable["Total Revenue"].where(usable["Total Revenue"] > 0)
            margin = usable[numerator] / revenue
            for period, value in zip(usable["Date"], margin):
                if pd.notna(value) and abs(value) > 1:
                    findings.append(_finding(
                        "warning", "extreme_margin", ticker,
                        f"{label} is {value:.0%}; larger than revenue.",
                        period, value,
                    ))

    if "Stockholders Equity" in usable.columns:
        for period, value in zip(usable["Date"], usable["Stockholders Equity"]):
            if pd.notna(value) and value <= 0:
                findings.append(_finding(
                    "warning", "negative_equity", ticker,
                    "Stockholders' equity is zero or negative. Debt_To_Equity "
                    "is left missing for this period and the model fills it "
                    "with the training median, which hides this risk signal.",
                    period, value,
                ))

    if {"Total Debt", "Total Assets"} <= set(usable.columns):
        over = usable[usable["Total Debt"] > usable["Total Assets"]]
        for period, value in zip(over["Date"], over["Total Debt"]):
            findings.append(_finding(
                "warning", "debt_above_assets", ticker,
                "Total debt is larger than total assets.", period, value,
            ))

    if "Net Income" in usable.columns and len(usable) > 1:
        income = usable.set_index("Date")["Net Income"].dropna()
        sign = np.sign(income)
        flips = sign[(sign != sign.shift()) & sign.shift().notna()]
        for period in flips.index:
            findings.append(_finding(
                "info", "net_income_sign_change", ticker,
                "Net income changed sign (profit/loss) from the previous "
                "period.", period, income.loc[period],
            ))

    # One-off events from unusual items in the raw income statement
    if {"Total Unusual Items", "Net Income"} <= set(usable.columns):
        for _, row in usable.iterrows():
            items, income = row["Total Unusual Items"], row["Net Income"]
            if pd.notna(items) and pd.notna(income) and income != 0:
                share = abs(items) / abs(income)
                if share > UNUSUAL_ITEMS_LIMIT:
                    findings.append(_finding(
                        "warning", "one_off_items", ticker,
                        f"Unusual items equal {share:.0%} of net income; "
                        "margins for this period may not reflect normal "
                        "operations.", row["Date"], items,
                    ))

    # Accounting identities (provider consistency).
    # Assets = liabilities + equity including minority interest. Use the raw
    # "Total Equity Gross Minority Interest" when available; otherwise
    # shareholders' equity, where a gap may just be minority interest.
    identity = {"Total Assets", "Total Liabilities Net Minority Interest"}
    if identity <= set(usable.columns):
        for _, row in usable.dropna(subset=list(identity)).iterrows():
            gross = row.get("Total Equity Gross Minority Interest")
            if pd.notna(gross):
                equity, includes_minority = gross, True
            elif pd.notna(row.get("Stockholders Equity")):
                equity, includes_minority = row["Stockholders Equity"], False
            else:
                continue
            gap = row["Total Assets"] - (
                row["Total Liabilities Net Minority Interest"] + equity
            )
            share = abs(gap) / abs(row["Total Assets"]) if row["Total Assets"] else 0
            if includes_minority and share > SMALL_TOLERANCE:
                findings.append(_finding(
                    "warning", "balance_sheet_identity", ticker,
                    "Assets differ from liabilities + total equity "
                    f"(including minority interest) by {share:.1%} of assets.",
                    row["Date"], gap,
                ))
            elif not includes_minority and share > IDENTITY_TOLERANCE:
                findings.append(_finding(
                    "info", "balance_sheet_identity", ticker,
                    f"Assets differ from liabilities + shareholders' equity by "
                    f"{share:.1%} of assets; minority interest is not "
                    "available to confirm the difference.",
                    row["Date"], gap,
                ))

    if {"Working Capital", "Current Assets", "Current Liabilities"} <= set(usable.columns):
        subset = usable.dropna(subset=["Working Capital", "Current Assets",
                                       "Current Liabilities"])
        for _, row in subset.iterrows():
            expected = row["Current Assets"] - row["Current Liabilities"]
            scale = max(abs(row["Current Assets"]), 1)
            if abs(row["Working Capital"] - expected) / scale > SMALL_TOLERANCE:
                findings.append(_finding(
                    "warning", "working_capital_identity", ticker,
                    "Working capital does not equal current assets minus "
                    "current liabilities.", row["Date"],
                    row["Working Capital"] - expected,
                ))

    if {"Free Cash Flow", "Operating Cash Flow", "Capital Expenditure"} <= set(usable.columns):
        subset = usable.dropna(subset=["Free Cash Flow", "Operating Cash Flow",
                                       "Capital Expenditure"])
        for _, row in subset.iterrows():
            expected = row["Operating Cash Flow"] + row["Capital Expenditure"]
            scale = max(abs(row["Operating Cash Flow"]), 1)
            if abs(row["Free Cash Flow"] - expected) / scale > SMALL_TOLERANCE:
                findings.append(_finding(
                    "warning", "free_cash_flow_identity", ticker,
                    "Free cash flow does not equal operating cash flow plus "
                    "capital expenditure.", row["Date"],
                    row["Free Cash Flow"] - expected,
                ))

    summary = {
        "ticker": ticker,
        "records": int(len(data)),
        "usable_quarters": int(len(dates)),
        "first_period": None if dates.empty else str(dates.iloc[0].date()),
        "latest_period": None if dates.empty else str(dates.iloc[-1].date()),
        "fiscal_quarter_end_months": [int(m) for m in months],
        "key_field_coverage": {
            field: (
                round(float(usable[field].notna().mean()), 3)
                if field in usable.columns and len(usable) else 0.0
            )
            for field in KEY_FIELDS
        },
        "present_key_fields": present,
    }
    return summary, findings


def audit_quarterly(df, as_of=None, currencies=None, raw_dir=None):
    """Audit a combined quarterly dataset (one row per company and period).

    Parameters
    ----------
    df : DataFrame with at least Date and Ticker columns.
    as_of : date used for the stale-data check (default: today).
    currencies : optional {ticker: reporting currency} from the provider.
    raw_dir : optional folder with raw *_income_statement.csv files, used
        to detect one-off (unusual) items.
    """
    as_of = pd.Timestamp(as_of or date.today())
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    findings = []
    invalid_dates = data["Date"].isna().sum()
    if invalid_dates:
        findings.append(_finding(
            "error", "invalid_date", None,
            f"{invalid_dates} record(s) have a missing or invalid date.",
        ))
    data = data.dropna(subset=["Date", "Ticker"])

    tickers = sorted(data["Ticker"].astype(str).unique())
    extras_all = load_raw_extras(raw_dir, tickers)

    companies = {}
    for ticker in tickers:
        extras = extras_all[extras_all["Ticker"] == ticker]
        summary, company_findings = _audit_company(
            ticker, data[data["Ticker"] == ticker], as_of, extras
        )
        companies[ticker] = summary
        findings.extend(company_findings)

    # Cross-company comparability
    currency_map = {}
    for ticker in tickers:
        if currencies and currencies.get(ticker):
            currency_map[ticker] = {"currency": currencies[ticker],
                                    "source": "provider"}
        else:
            currency_map[ticker] = {"currency": infer_currency(ticker),
                                    "source": "inferred from ticker suffix"}
        companies[ticker]["reporting_currency"] = currency_map[ticker]

    distinct_currencies = sorted({v["currency"] for v in currency_map.values()})
    if len(distinct_currencies) > 1:
        detail = ", ".join(
            f"{t}: {v['currency']}" for t, v in currency_map.items()
        )
        findings.append(_finding(
            "warning", "mixed_currencies", None,
            f"Companies report in different currencies ({detail}). Absolute "
            "amounts are not comparable across companies; ratios are.",
        ))

    calendars = {
        t: tuple(companies[t]["fiscal_quarter_end_months"]) for t in tickers
    }
    if len(set(calendars.values())) > 1:
        off_calendar = [
            t for t, months in calendars.items()
            if months and not set(months) <= {3, 6, 9, 12}
        ]
        if off_calendar:
            findings.append(_finding(
                "info", "mixed_fiscal_calendars", None,
                "Some companies' quarters end in different months "
                f"({', '.join(off_calendar)}). Rows with the same Date are "
                "not the same economic quarter, and date-based train/test "
                "splits mix calendars.",
            ))

    for ticker in tickers:
        coverage = companies[ticker]["key_field_coverage"]
        if (
            coverage.get("Current Assets", 0) == 0
            and coverage.get("Operating Income", 0) == 0
            and coverage.get("Total Revenue", 0) > 0
        ):
            findings.append(_finding(
                "info", "financial_company_structure", ticker,
                "No current assets or operating income reported; the "
                "statements look like a bank/financial company. Margin and "
                "liquidity ratios are not comparable with other companies.",
            ))

    findings.sort(key=lambda f: (
        SEVERITY_ORDER[f["severity"]], f["ticker"] or "", f["check"],
        f["period"] or "",
    ))

    counts = {
        severity: sum(1 for f in findings if f["severity"] == severity)
        for severity in SEVERITY_ORDER
    }
    return {
        "as_of": str(as_of.date()),
        "companies_audited": len(tickers),
        "records": int(len(data)),
        "finding_counts": counts,
        "companies": companies,
        "findings": findings,
    }
