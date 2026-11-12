"""Distress features and labels from an annual financial panel.

Input: one row per company and fiscal year with the canonical columns of
edgar_frames.CANONICAL (revenue, net_income, operating_income, ...). The
same code serves training (SEC EDGAR panel) and scoring one company (Yahoo
annual statements mapped by yahoo_annual_panel()).

Distress definition (a year is "in distress" if any holds):
  1. cash-burning loss:  net income < 0 and operating cash flow < 0
  2. negative equity:    shareholders' equity <= 0
  3. weak coverage:      operating income / interest expense < 1
                         (interest expense > 0.5% of assets)
Target: DISTRESS_NEXT = the company's next fiscal year (350-380 days later)
is in distress, i.e. distress within the next 12 months.

Features use only the current and previous fiscal year. Ratios are clipped
to fixed bounds so a few extreme values do not dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "distress_next"
GROUP = "cik"
YEAR_GAP = (350, 380)
AVAILABILITY_LAG_DAYS = 90   # latest 10-K deadline after fiscal year end

FEATURES = [
    # size and profitability
    "log_assets", "roa", "operating_margin", "net_margin", "gross_margin",
    # cash flow quality
    "ocf_to_assets", "ocf_to_liabilities", "accruals",
    # leverage and liquidity
    "leverage", "debt_to_assets", "equity_to_liabilities", "current_ratio",
    "cash_to_assets", "working_capital_to_assets", "retained_earnings_to_assets",
    "interest_coverage",
    # Altman Z'' (non-manufacturing, book-value version)
    "altman_z2",
    # one-year changes
    "revenue_growth", "d_roa", "d_leverage", "d_current_ratio",
    "d_gross_margin", "d_operating_margin",
    # current state
    "loss_now", "cash_burn_now", "negative_equity_now", "weak_coverage_now",
    "distress_now",
]

FEATURE_DESCRIPTIONS = {
    "log_assets": "Company size (log of total assets)",
    "roa": "Return on assets (net income / assets)",
    "operating_margin": "Operating margin",
    "net_margin": "Net margin",
    "gross_margin": "Gross margin",
    "ocf_to_assets": "Operating cash flow / assets",
    "ocf_to_liabilities": "Operating cash flow / liabilities",
    "accruals": "Accruals ((net income - operating cash flow) / assets)",
    "leverage": "Liabilities / assets",
    "debt_to_assets": "Long-term debt / assets",
    "equity_to_liabilities": "Equity / liabilities",
    "current_ratio": "Current ratio",
    "cash_to_assets": "Cash / assets",
    "working_capital_to_assets": "Working capital / assets",
    "retained_earnings_to_assets": "Retained earnings / assets",
    "interest_coverage": "Interest coverage (operating income / interest)",
    "altman_z2": "Altman Z'' score",
    "revenue_growth": "Revenue growth vs previous year",
    "d_roa": "Change in return on assets",
    "d_leverage": "Change in leverage",
    "d_current_ratio": "Change in current ratio",
    "d_gross_margin": "Change in gross margin",
    "d_operating_margin": "Change in operating margin",
    "loss_now": "Net loss this year",
    "cash_burn_now": "Loss and negative operating cash flow this year",
    "negative_equity_now": "Negative equity this year",
    "weak_coverage_now": "Interest coverage below 1 this year",
    "distress_now": "In distress this year",
}

CLIP = {
    "roa": (-2, 1), "operating_margin": (-5, 1), "net_margin": (-5, 1),
    "gross_margin": (-2, 1), "ocf_to_assets": (-2, 1), "ocf_to_liabilities": (-5, 5),
    "accruals": (-2, 2), "leverage": (0, 5), "debt_to_assets": (0, 3),
    "equity_to_liabilities": (-5, 20), "current_ratio": (0, 20),
    "cash_to_assets": (0, 1), "working_capital_to_assets": (-3, 1),
    "retained_earnings_to_assets": (-20, 2), "interest_coverage": (-20, 50),
    "altman_z2": (-30, 30), "revenue_growth": (-1, 5), "d_roa": (-2, 2),
    "d_leverage": (-3, 3), "d_current_ratio": (-10, 10), "d_gross_margin": (-2, 2),
    "d_operating_margin": (-5, 5),
}


def _div(a, b):
    b = b.where(b.abs() > 0)
    return a / b


def distress_flags(panel):
    """Per-row distress components and overall flag (NaN when not computable)."""
    ni, ocf = panel["net_income"], panel["operating_cash_flow"]
    equity = panel["equity"]
    interest = panel["interest_expense"]
    assets = panel["total_assets"]

    cash_burn = ((ni < 0) & (ocf < 0)).where(ni.notna() & ocf.notna())
    negative_equity = (equity <= 0).where(equity.notna())
    material_interest = interest > 0.005 * assets
    coverage = _div(panel["operating_income"], interest)
    weak = (material_interest & (coverage < 1)).where(
        interest.notna() & panel["operating_income"].notna())
    weak = weak.where(interest.notna(), 0.0)  # no interest reported -> not weak

    parts = pd.concat([cash_burn, negative_equity, weak], axis=1).astype(float)
    distress = parts.max(axis=1, skipna=True)
    distress[parts.isna().all(axis=1)] = np.nan
    return cash_burn.astype(float), negative_equity.astype(float), weak.astype(float), distress


def add_features(panel):
    """Add FEATURES (and distress flags) to a panel sorted by company/year."""
    p = panel.copy()
    p["fiscal_year_end"] = pd.to_datetime(p["fiscal_year_end"])
    p = p.sort_values([GROUP, "fiscal_year_end"]).reset_index(drop=True)

    if "total_liabilities" in p:
        derived = p["total_assets"] - p["equity"]
        p["total_liabilities"] = p["total_liabilities"].fillna(derived)

    a, l = p["total_assets"].where(p["total_assets"] > 0), p["total_liabilities"]
    rev = p["revenue"].where(p["revenue"] > 0)
    p["log_assets"] = np.log10(a)
    p["roa"] = p["net_income"] / a
    p["operating_margin"] = p["operating_income"] / rev
    p["net_margin"] = p["net_income"] / rev
    p["gross_margin"] = p["gross_profit"] / rev
    p["ocf_to_assets"] = p["operating_cash_flow"] / a
    p["ocf_to_liabilities"] = _div(p["operating_cash_flow"], l.where(l > 0))
    p["accruals"] = (p["net_income"] - p["operating_cash_flow"]) / a
    p["leverage"] = l / a
    p["debt_to_assets"] = p["long_term_debt"].fillna(0) / a
    p["equity_to_liabilities"] = _div(p["equity"], l.where(l > 0))
    p["current_ratio"] = _div(p["current_assets"], p["current_liabilities"].where(p["current_liabilities"] > 0))
    p["cash_to_assets"] = p["cash"] / a
    working_capital = p["current_assets"] - p["current_liabilities"]
    p["working_capital_to_assets"] = working_capital / a
    p["retained_earnings_to_assets"] = p["retained_earnings"] / a
    p["interest_coverage"] = _div(p["operating_income"], p["interest_expense"].where(p["interest_expense"] > 0))
    p["altman_z2"] = (
        6.56 * p["working_capital_to_assets"]
        + 3.26 * p["retained_earnings_to_assets"]
        + 6.72 * (p["operating_income"] / a)
        + 1.05 * p["equity_to_liabilities"]
    )

    cash_burn, negative_equity, weak, distress = distress_flags(p)
    p["loss_now"] = (p["net_income"] < 0).astype(float).where(p["net_income"].notna())
    p["cash_burn_now"], p["negative_equity_now"] = cash_burn, negative_equity
    p["weak_coverage_now"], p["distress_now"] = weak, distress

    previous = p.groupby(GROUP).shift(1)
    gap = (p["fiscal_year_end"] - previous["fiscal_year_end"]).dt.days
    consecutive = gap.between(*YEAR_GAP)
    prev_rev = previous["revenue"].where(previous["revenue"] > 0)
    p["revenue_growth"] = (p["revenue"] / prev_rev - 1).where(consecutive)
    for name in ["roa", "leverage", "current_ratio", "gross_margin", "operating_margin"]:
        p[f"d_{name}"] = (p[name] - previous[name]).where(consecutive)

    for name, (low, high) in CLIP.items():
        p[name] = p[name].replace([np.inf, -np.inf], np.nan).clip(low, high)
    return p


def add_labels(p):
    """Add DISTRESS_NEXT and timing columns (rows without a next year get NaN)."""
    nxt = p.groupby(GROUP).shift(-1)
    gap = (nxt["fiscal_year_end"] - p["fiscal_year_end"]).dt.days
    p[TARGET] = nxt["distress_now"].where(gap.between(*YEAR_GAP))
    p["available_date"] = p["fiscal_year_end"] + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
    p["label_available_date"] = nxt["fiscal_year_end"] + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
    p["year"] = p["fiscal_year_end"].dt.year
    return p


def build_training_set(panel, min_assets=10e6):
    """Operating companies with revenue, excluding banks/insurers (no current
    assets reported) and shells (assets below min_assets)."""
    p = add_labels(add_features(panel))
    keep = (
        p["current_assets"].notna()
        & (p["total_assets"] >= min_assets)
        & (p["revenue"] > 0)
        & p[TARGET].notna()
    )
    return p[keep].reset_index(drop=True)


YAHOO_MAP = {
    "revenue": ["Total Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "operating_income": ["Operating Income", "EBIT"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "operating_cash_flow": ["Operating Cash Flow"],
    "gross_profit": ["Gross Profit"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "equity": ["Stockholders Equity"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "retained_earnings": ["Retained Earnings"],
    "cash": ["Cash And Cash Equivalents"],
    "long_term_debt": ["Long Term Debt"],
}


def yahoo_annual_panel(income, balance, cashflow, key="company"):
    """Map yfinance annual statements (line items x dates) to the canonical
    panel. Note: Yahoo's definitions differ slightly from the SEC concepts
    the model was trained on (see the model report)."""
    tables = [t for t in (income, balance, cashflow) if t is not None and not t.empty]
    if not tables:
        return pd.DataFrame()
    dates = sorted({d for t in tables for d in t.columns})
    rows = []
    for date in dates:
        row = {GROUP: key, "fiscal_year_end": pd.Timestamp(date)}
        for name, items in YAHOO_MAP.items():
            value = np.nan
            for table in tables:
                for item in items:
                    if item in table.index and date in table.columns:
                        v = pd.to_numeric(table.loc[item, date], errors="coerce")
                        if isinstance(v, pd.Series):
                            v = v.iloc[0]
                        if pd.notna(v):
                            value = float(v)
                            break
                if pd.notna(value):
                    break
            row[name] = value
        rows.append(row)
    panel = pd.DataFrame(rows)
    panel["interest_expense"] = panel["interest_expense"].abs()
    return panel
