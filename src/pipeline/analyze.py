"""Analyse one company and produce the company_analysis/1.0 result.

Steps: resolve the ticker/name -> collect quarterly statements -> combine ->
ratios -> data-quality audit -> rule-based risk flags -> experimental ML
signal -> summary.json + report.md.

Outputs (outputs/<SYMBOL>_<timestamp>/):
  summary.json               the result; schema: schemas/company_analysis.schema.json
  report.md                  readable report generated from summary.json
  quarterly_financials.csv   statement values + ratios, one row per quarter
  data_quality.md            detailed data-quality report
  raw/                       statements exactly as downloaded
  pipeline.log               console output of the reused pipeline steps

Deterministic results (metrics, risk_flags) are kept separate from the
experimental ML output (ml_signal). The output contract is documented in
docs/output_schema.md.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_collection.collect_quarterly_data import statement_to_rows
from src.data_collection.prepare_quarterly_dataset import build_quarterly_dataset
from src.data_collection.provider import (
    STATEMENT_ATTRIBUTES,
    ProviderError,
    YahooFinanceProvider,
)
from src.data_collection.ticker_resolver import TickerResolutionError, resolve
from src.data_quality.audit import audit_quarterly
from src.data_quality.report import render_audit_markdown
from src.data_quality.verify_features import (
    RATIO_DEFINITIONS,
    verify_ratio_formulas,
)
from src.ml.ml_signal import build_ml_signal
from src.ml.prepare_revenue_decline_features import add_ratio_features
from src.pipeline.report import render_company_report
from src.pipeline.risk_flags import evaluate_risk_flags


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SCHEMA_VERSION = "company_analysis/1.0"

DISCLAIMER = (
    "Informational research output only; not investment advice. Figures come "
    "from a third-party data provider and may be incomplete, delayed, or "
    "restated. Risk flags are general screening rules, and the ML signal is "
    "experimental and has not been shown to predict better than chance."
)

PERIOD_FIELDS = {
    "Total Revenue": "revenue",
    "Net Income": "net_income",
    "Operating Income": "operating_income",
    "EBIT": "ebit",
    "Interest Expense": "interest_expense",
    "Total Assets": "total_assets",
    "Total Debt": "total_debt",
    "Stockholders Equity": "stockholders_equity",
    "Cash And Cash Equivalents": "cash_and_equivalents",
    "Current Assets": "current_assets",
    "Current Liabilities": "current_liabilities",
    "Operating Cash Flow": "operating_cash_flow",
    "Free Cash Flow": "free_cash_flow",
}
RATIO_FIELDS = {
    "Net_Income_Margin": "net_income_margin",
    "Operating_Margin": "operating_margin",
    "Debt_To_Assets": "debt_to_assets",
    "Debt_To_Equity": "debt_to_equity",
    "Current_Ratio": "current_ratio",
}

# Data-quality checks that are surfaced as top-level warnings.
SURFACED_CHECKS = {
    "stale_data", "incomplete_latest_period", "quarterly_gap",
    "negative_equity", "duplicate_period", "one_off_items", "revenue_jump",
    "mixed_currencies", "financial_company_structure",
}


def _clean(value):
    """JSON-safe value: NaN/inf -> None, numpy numbers -> float."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return None if math.isnan(number) or math.isinf(number) else number


def _slug(text):
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", text).strip("_")[:40] or "input"


def _rel(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _periods(table):
    """One dict per usable quarter (oldest first) with values and ratios."""
    income_cols = [c for c in ("Total Revenue", "Net Income") if c in table]
    usable = table.dropna(subset=income_cols, how="all") if income_cols else table
    usable = usable.sort_values("Date").reset_index(drop=True)

    revenue = usable.get("Total Revenue")
    periods = []
    for i, row in usable.iterrows():
        values = {
            key: _clean(row.get(column)) for column, key in PERIOD_FIELDS.items()
        }
        ratios = {
            key: _clean(row.get(column)) for column, key in RATIO_FIELDS.items()
        }
        changes = {"revenue_change_qoq": None, "revenue_change_yoy": None}
        if revenue is not None:
            current = row.get("Total Revenue")
            date = row["Date"]
            if i > 0:
                previous = usable.loc[i - 1]
                gap = (date - previous["Date"]).days
                if 70 <= gap <= 110 and previous["Total Revenue"] and \
                        pd.notna(previous["Total Revenue"]) and pd.notna(current):
                    changes["revenue_change_qoq"] = _clean(
                        current / previous["Total Revenue"] - 1
                    )
            # Same quarter one year earlier (350-380 days back).
            earlier = usable[
                ((date - usable["Date"]).dt.days.between(350, 380))
            ]
            if not earlier.empty and pd.notna(current):
                base = earlier.iloc[0]["Total Revenue"]
                if pd.notna(base) and base > 0:
                    changes["revenue_change_yoy"] = _clean(current / base - 1)
        periods.append({
            "period_end": str(row["Date"].date()),
            "values": values,
            "ratios": ratios,
            "changes": changes,
        })
    return periods, usable


def _new_payload(user_input, prefer_indian, as_of):
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request": {
            "input": user_input,
            "prefer_indian": bool(prefer_indian),
            "as_of": None if as_of is None else str(as_of),
        },
        "company": None,
        "source": None,
        "coverage": None,
        "metrics": None,
        "risk_flags": None,
        "ml_signal": None,
        "data_quality": None,
        "warnings": [],
        "error": None,
        "disclaimer": DISCLAIMER,
        "files": {},
    }


def analyze_company(user_input, output_root=None, provider=None, as_of=None,
                    prefer_indian=False, echo=print, training_file=None,
                    evaluation_file=None):
    """Run the single-company pipeline. Always writes summary.json and
    returns the payload (status "ok" or "error")."""
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = _new_payload(user_input, prefer_indian, as_of)
    warn = lambda code, message: payload["warnings"].append(  # noqa: E731
        {"code": code, "message": message})

    def finish(run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        report_path = run_dir / "report.md"
        payload["files"]["summary"] = _rel(summary_path)
        payload["files"]["report"] = _rel(report_path)
        summary_path.write_text(json.dumps(payload, indent=2, default=str),
                                encoding="utf-8")
        report_path.write_text(render_company_report(payload), encoding="utf-8")
        return payload

    def fail(code, message, run_dir, candidates=None):
        payload["status"] = "error"
        payload["error"] = {"code": code, "message": message,
                            "candidates": candidates or []}
        return finish(run_dir)

    error_dir = output_root / f"{_slug(user_input)}_{stamp}"
    try:
        provider = provider or YahooFinanceProvider()
        echo(f"Resolving '{user_input}'...")
        resolution = resolve(user_input, provider, prefer_indian=prefer_indian)
    except TickerResolutionError as exc:
        return fail(exc.code, exc.message, error_dir, exc.candidates)
    except ProviderError as exc:
        return fail(exc.code, str(exc), error_dir)
    except Exception as exc:  # never return a non-JSON failure
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return fail("internal_error", f"{type(exc).__name__}: {exc}", error_dir)

    run_dir = output_root / f"{_slug(resolution.symbol)}_{stamp}"
    try:
        return _analyze_resolved(resolution, provider, run_dir, payload, warn,
                                 finish, fail, as_of, echo, training_file,
                                 evaluation_file)
    except ProviderError as exc:
        return fail(exc.code, str(exc), run_dir)
    except Exception as exc:  # never return a non-JSON failure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return fail("internal_error", f"{type(exc).__name__}: {exc}", run_dir)


def _analyze_resolved(resolution, provider, run_dir, payload, warn, finish, fail,
                      as_of, echo, training_file, evaluation_file):
    symbol = resolution.symbol
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    method_codes = {"name_search": "name_search_match",
                    "suffix": "exchange_suffix_added"}
    for note in resolution.notes:
        code = ("other_listings" if note.startswith("Other listings")
                else method_codes.get(resolution.method, "resolution_note"))
        warn(code, note)
    echo(f"Resolved to {symbol} ({resolution.method}). Collecting statements...")

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    profile = provider.profile(symbol)
    statements = {}
    for statement in STATEMENT_ATTRIBUTES:
        table = provider.quarterly_statement(symbol, statement)
        if table.empty:
            statements[statement] = {"available": False, "periods": 0}
            warn("missing_statement",
                 f"The provider has no quarterly {statement.replace('_', ' ')} "
                 f"for {symbol}.")
            continue
        rows = statement_to_rows(table, symbol)
        rows.to_csv(raw_dir / f"{symbol}_{statement}.csv", index=False)
        dates = pd.to_datetime(rows["Date"], errors="coerce")
        statements[statement] = {
            "available": True,
            "periods": int(len(rows)),
            "latest_period": str(dates.max().date()) if dates.notna().any() else None,
        }

    payload["company"] = {
        "symbol": symbol,
        "name": profile.get("longName") or profile.get("shortName") or resolution.name,
        "exchange": profile.get("fullExchangeName") or profile.get("exchange")
        or resolution.exchange,
        "country": profile.get("country"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "reporting_currency": profile.get("financialCurrency"),
        "trading_currency": profile.get("currency"),
        "resolution": {
            "method": resolution.method,
            "alternatives": resolution.alternatives,
        },
    }
    payload["source"] = {
        "provider": provider.name,
        "retrieved_at": retrieved_at,
        "frequency": "quarterly",
        "statements": statements,
        "note": "Values are as reported by the provider, in the reporting "
                "currency's units (not thousands). Period end = fiscal quarter "
                "end, not the filing date.",
    }
    if not profile:
        warn("profile_unavailable",
             "Company profile (name, currency, sector) could not be fetched; "
             "the reporting currency is unknown.")

    log_file = run_dir / "pipeline.log"
    with open(log_file, "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log):
        combined = build_quarterly_dataset(raw_dir, tickers=[symbol])
    payload["files"]["pipeline_log"] = _rel(log_file)

    if combined is None or combined.empty:
        return fail("no_financial_data",
                    f"No usable quarterly income statement for {symbol}.", run_dir)

    table = add_ratio_features(combined)
    table_file = run_dir / "quarterly_financials.csv"
    table.to_csv(table_file, index=False)
    payload["files"]["quarterly_financials"] = _rel(table_file)

    currencies = {symbol: profile["financialCurrency"]} \
        if profile.get("financialCurrency") else None
    audit = audit_quarterly(combined, as_of=as_of, currencies=currencies,
                            raw_dir=raw_dir)
    ratio_check = verify_ratio_formulas(table)
    quality_file = run_dir / "data_quality.md"
    quality_file.write_text(
        render_audit_markdown(audit, ratio_check, title=f"Data quality: {symbol}"),
        encoding="utf-8",
    )
    payload["files"]["data_quality_report"] = _rel(quality_file)

    info = audit["companies"].get(symbol, {})
    payload["coverage"] = {
        "usable_quarters": info.get("usable_quarters"),
        "first_period": info.get("first_period"),
        "latest_period": info.get("latest_period"),
        "fiscal_quarter_end_months": info.get("fiscal_quarter_end_months"),
        "key_field_coverage": info.get("key_field_coverage"),
    }

    periods, usable = _periods(table)
    latest = periods[-1] if periods else None
    payload["metrics"] = {
        "currency": profile.get("financialCurrency"),
        "latest_period": latest["period_end"] if latest else None,
        "latest": None if latest is None else {
            "values": latest["values"],
            "ratios": latest["ratios"],
            "changes": latest["changes"],
        },
        "periods": periods,
        "ratio_definitions": {
            RATIO_FIELDS[name]: f"{num} / {den} (missing if {den} <= 0)"
            for name, (num, den) in RATIO_DEFINITIONS.items()
        },
    }

    financial_company = any(
        f["check"] == "financial_company_structure" and f["ticker"] == symbol
        for f in audit["findings"]
    )
    payload["risk_flags"] = evaluate_risk_flags(usable, financial_company)

    signal_kwargs = {}
    if training_file is not None:
        signal_kwargs["training_file"] = training_file
    if evaluation_file is not None:
        signal_kwargs["evaluation_file"] = evaluation_file
    latest_row = usable.iloc[-1] if not usable.empty else {}
    payload["ml_signal"] = build_ml_signal(
        {k: latest_row.get(k) for k in RATIO_FIELDS},
        latest["period_end"] if latest else None,
        financial_company=financial_company,
        **signal_kwargs,
    )

    payload["data_quality"] = {
        "finding_counts": audit["finding_counts"],
        "findings": audit["findings"],
        "ratio_formula_check": {
            k: v["status"] for k, v in ratio_check["features"].items()
        },
    }
    for finding in audit["findings"]:
        if finding["check"] in SURFACED_CHECKS and finding["severity"] != "info":
            warn(finding["check"], finding["message"])
        elif finding["check"] == "financial_company_structure":
            warn(finding["check"], finding["message"])

    echo(f"Done: {len(periods)} quarters, "
         f"{payload['risk_flags']['summary']['triggered']} risk flags triggered.")
    return finish(run_dir)
