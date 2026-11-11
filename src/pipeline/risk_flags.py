"""Rule-based (deterministic) financial risk flags.

These are plain screening rules on reported numbers, kept separate from the
experimental ML signal. Each flag reports:

  status     triggered | not_triggered | not_evaluable
  severity   high | medium | low (the rule's severity if triggered)
  value, threshold, period_end, inputs, and a plain-language explanation

Each rule uses the most recent quarter in which all of its inputs are
reported (some companies publish balance sheets only half-yearly), and says
so when that is older than the latest quarter.

Thresholds are common, general-purpose screening levels. They are not
tuned to any industry and a triggered flag is a prompt to look closer, not
a judgement about the company.
"""

from __future__ import annotations

import math

import pandas as pd


RULES_VERSION = "1.0"

THRESHOLDS = {
    "revenue_decline_yoy": -0.05,        # revenue more than 5% below last year
    "operating_margin_drop_yoy": -0.05,  # 5 percentage points lower
    "current_ratio": 1.0,
    "debt_to_equity": 2.0,
    "debt_to_assets": 0.5,
    "interest_coverage": 1.5,
}


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _latest_with(table, columns):
    """Most recent row where every column has a value, or None."""
    subset = table.dropna(subset=[c for c in columns if c in table.columns])
    if any(c not in table.columns for c in columns) or subset.empty:
        return None
    return subset.iloc[-1]


def _year_earlier(table, row, columns):
    """Row for the same quarter one year before `row` (350-380 days)."""
    days = (row["Date"] - table["Date"]).dt.days
    earlier = table[days.between(350, 380)].dropna(
        subset=[c for c in columns if c in table.columns]
    )
    return None if earlier.empty else earlier.iloc[-1]


def _flag(rule_id, category, severity, title, rule_text):
    return {
        "id": rule_id,
        "category": category,
        "severity": severity,
        "title": title,
        "rule": rule_text,
        "status": "not_evaluable",
        "value": None,
        "threshold": None,
        "period_end": None,
        "inputs": {},
        "explanation": "",
    }


def _set(flag, triggered, value, threshold, row, inputs, latest_date,
         when_true, when_false):
    flag["status"] = "triggered" if triggered else "not_triggered"
    flag["value"] = None if value is None else round(value, 6)
    flag["threshold"] = threshold
    flag["period_end"] = str(row["Date"].date())
    flag["inputs"] = {k: _num(v) for k, v in inputs.items()}
    flag["explanation"] = when_true if triggered else when_false
    if row["Date"] < latest_date:
        flag["explanation"] += (
            f" Based on {row['Date'].date()}, the latest quarter with the "
            "required figures."
        )
    return flag


def _not_evaluable(flag, reason):
    flag["status"] = "not_evaluable"
    flag["explanation"] = reason
    return flag


def evaluate_risk_flags(table, financial_company=False):
    """Evaluate every rule on one company's quarterly table.

    table: one row per quarter with Date and the statement columns
    (Total Revenue, Net Income, ...). Ratio columns are recomputed here.
    financial_company: True for bank-like statements, where liquidity and
    margin rules do not apply.
    """
    table = table.copy()
    table["Date"] = pd.to_datetime(table["Date"])
    table = table.sort_values("Date").reset_index(drop=True)
    latest_date = table["Date"].max()
    flags = []

    # Profitability ------------------------------------------------------
    flag = _flag("net_loss", "profitability", "medium", "Net loss",
                 "Net Income < 0 in the latest quarter")
    row = _latest_with(table, ["Net Income"])
    if row is None:
        flags.append(_not_evaluable(flag, "Net income is not reported."))
    else:
        income = row["Net Income"]
        flags.append(_set(
            flag, income < 0, income, 0, row, {"Net Income": income}, latest_date,
            "The company reported a net loss for the quarter.",
            "The company reported a net profit for the quarter.",
        ))

    flag = _flag("operating_loss", "profitability", "medium", "Operating loss",
                 "Operating Income < 0 in the latest quarter")
    row = _latest_with(table, ["Operating Income"])
    if financial_company:
        flags.append(_not_evaluable(
            flag, "Operating income is not reported for bank-like statements."))
    elif row is None:
        flags.append(_not_evaluable(flag, "Operating income is not reported."))
    else:
        value = row["Operating Income"]
        flags.append(_set(
            flag, value < 0, value, 0, row, {"Operating Income": value}, latest_date,
            "Core operations lost money in the quarter.",
            "Core operations were profitable in the quarter.",
        ))

    # Growth (year over year, so seasonality cancels out) -------------------
    flag = _flag("revenue_decline_yoy", "growth", "medium",
                 "Revenue down year over year",
                 "Revenue vs the same quarter a year earlier < -5%")
    row = _latest_with(table, ["Total Revenue"])
    earlier = None if row is None else _year_earlier(table, row, ["Total Revenue"])
    if row is None or earlier is None or not earlier["Total Revenue"] > 0:
        flags.append(_not_evaluable(
            flag, "Needs revenue for the same quarter one year earlier, which "
                  "is not in the available data."))
    else:
        change = row["Total Revenue"] / earlier["Total Revenue"] - 1
        flags.append(_set(
            flag, change < THRESHOLDS["revenue_decline_yoy"], change,
            THRESHOLDS["revenue_decline_yoy"], row,
            {"Total Revenue": row["Total Revenue"],
             "Total Revenue (year earlier)": earlier["Total Revenue"]},
            latest_date,
            f"Revenue was {change:+.1%} versus {earlier['Date'].date()}.",
            f"Revenue was {change:+.1%} versus {earlier['Date'].date()}.",
        ))

    flag = _flag("operating_margin_drop_yoy", "profitability", "low",
                 "Operating margin down year over year",
                 "Operating margin vs the same quarter a year earlier "
                 "< -5 percentage points")
    columns = ["Operating Income", "Total Revenue"]
    row = _latest_with(table, columns)
    earlier = None if row is None else _year_earlier(table, row, columns)
    if financial_company:
        flags.append(_not_evaluable(
            flag, "Operating margin is not meaningful for bank-like statements."))
    elif (row is None or earlier is None or not row["Total Revenue"] > 0
          or not earlier["Total Revenue"] > 0):
        flags.append(_not_evaluable(
            flag, "Needs operating income and revenue for the same quarter "
                  "one year earlier."))
    else:
        now = row["Operating Income"] / row["Total Revenue"]
        before = earlier["Operating Income"] / earlier["Total Revenue"]
        change = now - before
        flags.append(_set(
            flag, change < THRESHOLDS["operating_margin_drop_yoy"], change,
            THRESHOLDS["operating_margin_drop_yoy"], row,
            {"Operating margin": now, "Operating margin (year earlier)": before},
            latest_date,
            f"Operating margin fell from {before:.1%} to {now:.1%}.",
            f"Operating margin moved from {before:.1%} to {now:.1%}.",
        ))

    # Liquidity ------------------------------------------------------------
    flag = _flag("low_current_ratio", "liquidity", "medium",
                 "Current ratio below 1",
                 "Current Assets / Current Liabilities < 1.0")
    row = _latest_with(table, ["Current Assets", "Current Liabilities"])
    if financial_company:
        flags.append(_not_evaluable(
            flag, "Banks do not report current assets/liabilities."))
    elif row is None or not row["Current Liabilities"] > 0:
        flags.append(_not_evaluable(
            flag, "Current assets or current liabilities are not reported."))
    else:
        ratio = row["Current Assets"] / row["Current Liabilities"]
        flags.append(_set(
            flag, ratio < THRESHOLDS["current_ratio"], ratio,
            THRESHOLDS["current_ratio"], row,
            {"Current Assets": row["Current Assets"],
             "Current Liabilities": row["Current Liabilities"]},
            latest_date,
            f"Short-term liabilities exceed short-term assets (ratio "
            f"{ratio:.2f}). Some large companies run this way by design; "
            "check cash flow before reading it as stress.",
            f"Short-term assets cover short-term liabilities (ratio {ratio:.2f}).",
        ))

    # Leverage and solvency -------------------------------------------------
    flag = _flag("negative_equity", "solvency", "high",
                 "Negative shareholders' equity",
                 "Stockholders Equity <= 0")
    row = _latest_with(table, ["Stockholders Equity"])
    if row is None:
        flags.append(_not_evaluable(flag, "Shareholders' equity is not reported."))
    else:
        equity = row["Stockholders Equity"]
        flags.append(_set(
            flag, equity <= 0, equity, 0, row, {"Stockholders Equity": equity},
            latest_date,
            "Liabilities exceed assets on the balance sheet.",
            "Shareholders' equity is positive.",
        ))

    flag = _flag("high_debt_to_equity", "leverage", "medium",
                 "High debt to equity", "Total Debt / Stockholders Equity > 2.0")
    row = _latest_with(table, ["Total Debt", "Stockholders Equity"])
    if row is None:
        flags.append(_not_evaluable(flag, "Total debt or equity is not reported."))
    elif not row["Stockholders Equity"] > 0:
        flags.append(_not_evaluable(
            flag, "Equity is zero or negative, so the ratio is not meaningful "
                  "(see the negative-equity flag)."))
    else:
        ratio = row["Total Debt"] / row["Stockholders Equity"]
        flags.append(_set(
            flag, ratio > THRESHOLDS["debt_to_equity"], ratio,
            THRESHOLDS["debt_to_equity"], row,
            {"Total Debt": row["Total Debt"],
             "Stockholders Equity": row["Stockholders Equity"]},
            latest_date,
            f"Debt is {ratio:.2f} times shareholders' equity.",
            f"Debt is {ratio:.2f} times shareholders' equity.",
        ))

    flag = _flag("high_debt_to_assets", "leverage", "medium",
                 "High debt to assets", "Total Debt / Total Assets > 0.5")
    row = _latest_with(table, ["Total Debt", "Total Assets"])
    if row is None or not row["Total Assets"] > 0:
        flags.append(_not_evaluable(flag, "Total debt or assets are not reported."))
    else:
        ratio = row["Total Debt"] / row["Total Assets"]
        flags.append(_set(
            flag, ratio > THRESHOLDS["debt_to_assets"], ratio,
            THRESHOLDS["debt_to_assets"], row,
            {"Total Debt": row["Total Debt"], "Total Assets": row["Total Assets"]},
            latest_date,
            f"Debt finances {ratio:.0%} of total assets.",
            f"Debt finances {ratio:.0%} of total assets.",
        ))

    flag = _flag("low_interest_coverage", "solvency", "medium",
                 "Low interest coverage", "EBIT / Interest Expense < 1.5")
    row = _latest_with(table, ["EBIT", "Interest Expense"])
    if row is None:
        flags.append(_not_evaluable(flag, "EBIT or interest expense is not reported."))
    elif not abs(row["Interest Expense"]) > 0:
        flags.append(_not_evaluable(flag, "No interest expense reported."))
    else:
        coverage = row["EBIT"] / abs(row["Interest Expense"])
        flags.append(_set(
            flag, coverage < THRESHOLDS["interest_coverage"], coverage,
            THRESHOLDS["interest_coverage"], row,
            {"EBIT": row["EBIT"], "Interest Expense": row["Interest Expense"]},
            latest_date,
            f"Operating earnings cover interest only {coverage:.1f} times.",
            f"Operating earnings cover interest {coverage:.1f} times.",
        ))

    # Cash flow -------------------------------------------------------------
    flag = _flag("negative_operating_cash_flow", "cash_flow", "medium",
                 "Negative operating cash flow", "Operating Cash Flow < 0")
    row = _latest_with(table, ["Operating Cash Flow"])
    if row is None:
        flags.append(_not_evaluable(flag, "Quarterly cash flow is not reported."))
    else:
        value = row["Operating Cash Flow"]
        flags.append(_set(
            flag, value < 0, value, 0, row, {"Operating Cash Flow": value},
            latest_date,
            "Operations consumed cash in the quarter.",
            "Operations generated cash in the quarter.",
        ))

    flag = _flag("negative_free_cash_flow", "cash_flow", "low",
                 "Negative free cash flow", "Free Cash Flow < 0")
    row = _latest_with(table, ["Free Cash Flow"])
    if row is None:
        flags.append(_not_evaluable(flag, "Quarterly cash flow is not reported."))
    else:
        value = row["Free Cash Flow"]
        flags.append(_set(
            flag, value < 0, value, 0, row, {"Free Cash Flow": value}, latest_date,
            "Cash from operations did not cover capital spending in the "
            "quarter (often normal during heavy investment).",
            "Cash from operations covered capital spending in the quarter.",
        ))

    summary = {
        "rules_version": RULES_VERSION,
        "evaluated": sum(f["status"] != "not_evaluable" for f in flags),
        "not_evaluable": sum(f["status"] == "not_evaluable" for f in flags),
        "triggered": sum(f["status"] == "triggered" for f in flags),
        "triggered_by_severity": {
            severity: sum(
                f["status"] == "triggered" and f["severity"] == severity
                for f in flags
            )
            for severity in ("high", "medium", "low")
        },
    }
    return {"summary": summary, "flags": flags}
