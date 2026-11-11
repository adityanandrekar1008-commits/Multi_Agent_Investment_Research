"""Independent checks of the model's ratio features and target.

1. Ratio formulas: every ratio is recomputed with plain Python arithmetic
   (not with the pipeline's pandas code) and compared with the stored value.
2. Feature timing: model features must be built only from the same row's
   current-period statement values, never from future/outcome columns.
3. Target: Next_Period_Revenue_Decline is recomputed from the sequence of
   reported revenues to confirm the horizon (next reported quarter).
"""

from __future__ import annotations

import math

import pandas as pd


TARGET = "Next_Period_Revenue_Decline"

# feature -> (numerator column, denominator column)
# Every ratio is left missing when the denominator is missing or <= 0.
RATIO_DEFINITIONS = {
    "Net_Income_Margin": ("Net Income", "Total Revenue"),
    "Operating_Margin": ("Operating Income", "Total Revenue"),
    "Debt_To_Assets": ("Total Debt", "Total Assets"),
    "Debt_To_Equity": ("Total Debt", "Stockholders Equity"),
    "Current_Ratio": ("Current Assets", "Current Liabilities"),
}

# Columns that describe the future and must never be model inputs.
OUTCOME_COLUMNS = {
    TARGET,
    "Next_Date",
    "Next_Revenue",
    "Days_To_Next_Report",
    "Next_Revenue_Change_Percent",
}

# Identifiers that must not be model inputs either.
IDENTIFIER_COLUMNS = {"Date", "Ticker"}


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def hand_ratio(numerator, denominator):
    """Reference implementation: numerator / denominator, or None when the
    denominator is missing, zero, or negative, or the numerator is missing."""
    numerator, denominator = _number(numerator), _number(denominator)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def verify_ratio_formulas(features, tolerance=1e-9):
    """Compare stored ratio values with the reference implementation.

    Returns a dict with per-feature counts and a list of mismatches.
    """
    results = {}
    mismatches = []

    for feature, (num_col, den_col) in RATIO_DEFINITIONS.items():
        if feature not in features.columns:
            results[feature] = {"status": "feature column missing"}
            continue
        if num_col not in features.columns or den_col not in features.columns:
            results[feature] = {
                "status": "source column missing",
                "formula": f"{num_col} / {den_col}",
            }
            continue

        checked = matched = 0
        for _, row in features.iterrows():
            expected = hand_ratio(row[num_col], row[den_col])
            stored = _number(row[feature])
            checked += 1
            if expected is None and stored is None:
                matched += 1
            elif (
                expected is not None and stored is not None
                and math.isclose(expected, stored, rel_tol=tolerance,
                                 abs_tol=tolerance)
            ):
                matched += 1
            else:
                mismatches.append({
                    "feature": feature,
                    "ticker": row.get("Ticker"),
                    "date": str(row.get("Date"))[:10],
                    "stored": stored,
                    "expected": expected,
                })

        missing = int(features[feature].isna().sum())
        results[feature] = {
            "status": "ok" if matched == checked else "mismatch",
            "formula": f"{num_col} / {den_col} (missing if {den_col} <= 0)",
            "rows_checked": checked,
            "rows_matching": matched,
            "missing_values": missing,
        }

    return {"features": results, "mismatches": mismatches}


def check_feature_timing(feature_lists):
    """Check that no model feature is an outcome or identifier column, and
    that every ratio feature is built only from current-period columns.

    feature_lists : {name: [feature, ...]} collected from the ML scripts.
    """
    problems = []
    for name, features in feature_lists.items():
        for feature in features:
            if feature in OUTCOME_COLUMNS:
                problems.append(
                    f"{name}: '{feature}' is a future/outcome column."
                )
            elif feature in IDENTIFIER_COLUMNS:
                problems.append(f"{name}: '{feature}' is an identifier.")
            elif feature.startswith("Next_") or "Future" in feature:
                problems.append(
                    f"{name}: '{feature}' looks like a future-period column."
                )
            elif feature not in RATIO_DEFINITIONS:
                problems.append(
                    f"{name}: '{feature}' has no documented definition; "
                    "check that it uses current-period data only."
                )
    return problems


def verify_target(base, labelled, gap_days=(70, 110)):
    """Recompute the revenue-decline target from the base quarterly dataset
    and compare it with the labelled dataset.

    base : dataset before labelling (Date, Ticker, Total Revenue)
    labelled : dataset with TARGET
    """
    base = base.copy()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    expected = {}

    for ticker, group in base.sort_values("Date").groupby("Ticker"):
        rows = list(group[["Date", "Total Revenue"]].itertuples(index=False))
        for current, following in zip(rows[:-1], rows[1:]):
            revenue, next_revenue = _number(current[1]), _number(following[1])
            days = (following[0] - current[0]).days
            if (
                revenue is None or next_revenue is None or revenue <= 0
                or not gap_days[0] <= days <= gap_days[1]
            ):
                continue
            expected[(ticker, current[0])] = int(next_revenue < revenue)

    labelled = labelled.copy()
    labelled["Date"] = pd.to_datetime(labelled["Date"], errors="coerce")
    stored = {
        (row.Ticker, row.Date): int(getattr(row, TARGET))
        for row in labelled[["Ticker", "Date", TARGET]].itertuples(index=False)
    }

    disagreements = [
        {"ticker": key[0], "date": str(key[1].date()),
         "expected": expected.get(key), "stored": stored.get(key)}
        for key in sorted(set(expected) | set(stored),
                          key=lambda k: (k[0], k[1]))
        if expected.get(key) != stored.get(key)
    ]

    return {
        "definition": (
            "1 if the next reported quarter's Total Revenue is lower than the "
            "current quarter's; only transitions 70-110 days apart are "
            "labelled."
        ),
        "horizon": "next reported quarter (about 3 months)",
        "expected_labels": len(expected),
        "stored_labels": len(stored),
        "disagreements": disagreements,
    }
