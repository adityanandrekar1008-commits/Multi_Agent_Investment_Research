"""Quantitative Risk Agent.

Combines:
  - the deterministic analysis (company_analysis/1.0: quarterly metrics,
    11 rule-based risk flags, data-quality findings);
  - the financial-distress model (trained on SEC EDGAR data, see
    reports/distress/) scored on the company's latest annual statements,
    explained with SHAP: which factors push the risk up or down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.common import num, result, signal
from src.ml.distress import model as distress_model
from src.ml.distress.features import FEATURES, add_features, yahoo_annual_panel
from src.pipeline.analyze import analyze_company


FLAG_SEVERITY = {"high": "high", "medium": "medium", "low": "low"}


def usd_rate(provider, currency):
    """Latest <currency>/USD rate from Yahoo (e.g. INRUSD=X), or None."""
    try:
        prices = provider.price_history(f"{currency.upper()}USD=X", period="1mo")
        return float(prices["Close"].dropna().iloc[-1]) if not prices.empty else None
    except Exception:
        return None


def distress_assessment(symbol, provider, financial_company, bundle=None, currency="USD"):
    bundle = bundle if bundle is not None else distress_model.load_model()
    if bundle is None:
        return {"status": "unavailable",
                "reason": "Distress model not trained yet (run `python main.py distress`)."}
    if financial_company:
        return {"status": "unavailable",
                "reason": "The distress model excludes banks and insurers."}
    income, balance, cashflow = provider.annual_statements(symbol)
    panel = yahoo_annual_panel(income, balance, cashflow, key=symbol)
    if panel.empty:
        return {"status": "unavailable", "reason": "No annual statements available."}
    features = add_features(panel)
    latest = features.dropna(subset=["total_assets"]).iloc[-1] if features["total_assets"].notna().any() else None
    if latest is None:
        return {"status": "unavailable", "reason": "Annual statements lack total assets."}
    row = {f: num(latest.get(f)) for f in FEATURES}
    # The model learned company size in US dollars; ratios are currency-free.
    fx_note = None
    if currency and currency.upper() != "USD" and row.get("log_assets") is not None:
        rate = usd_rate(provider, currency)
        if rate:
            row["log_assets"] = float(np.log10(10 ** row["log_assets"] * rate))
            fx_note = f"Total assets converted from {currency} to USD at {rate:.6f} for the size factor."
        else:
            row["log_assets"] = None
            fx_note = f"No {currency}/USD rate available; company size left out (filled by the model)."
    known = sum(v is not None for v in row.values())
    if known < len(FEATURES) // 2:
        return {"status": "unavailable",
                "reason": f"Only {known} of {len(FEATURES)} model inputs available."}
    meta = bundle["meta"]
    explanation = distress_model.explain_row(
        bundle, {k: (np.nan if v is None else v) for k, v in row.items()},
        background=meta.get("background_sample"))
    evaluation = meta.get("evaluation", {})
    return {
        "status": "available",
        "model": meta["model_name"],
        "fiscal_year_end": str(pd.Timestamp(latest["fiscal_year_end"]).date()),
        "score": explanation["score"],
        "riskier_than_percent_of_training": (
            int(explanation["riskier_than_percent_of_training"])
            if explanation["riskier_than_percent_of_training"] else 0),
        "in_distress_now": bool(row.get("distress_now") == 1),
        "top_factors": [
            {**c, "effect": "raises risk" if c["shap"] > 0 else "lowers risk"}
            for c in explanation["contributions"][:6]
        ],
        "inputs_available": known,
        "currency_note": fx_note,
        "model_quality": {
            "roc_auc": evaluation.get("roc_auc"),
            "pr_auc": evaluation.get("pr_auc"),
            "base_rate": evaluation.get("base_rate"),
            "validation": "later years and companies not seen in training",
        },
        "notes": [
            "Distress = losses with negative operating cash flow, negative equity, or "
            "operating income below interest expense, within the next fiscal year.",
            "Trained on US SEC filers; statements here come from Yahoo Finance, whose "
            "definitions differ slightly. Scores are not calibrated probabilities.",
        ],
    }


def run(symbol, provider, output_root=None, analysis=None, bundle=None):
    analysis = analysis or analyze_company(symbol, output_root=output_root,
                                           provider=provider, echo=lambda *a: None)
    if analysis["status"] != "ok":
        return result("quantitative_risk", "error", analysis["error"]["message"],
                      errors=[analysis["error"]["code"]]), analysis

    financial = any(w["code"] == "financial_company_structure" for w in analysis["warnings"])
    signals = []
    for flag in analysis["risk_flags"]["flags"]:
        if flag["status"] == "triggered":
            signals.append(signal(f"flag_{flag['id']}", FLAG_SEVERITY[flag["severity"]],
                                  "negative", flag["title"], flag["explanation"]))

    try:
        currency = (analysis.get("company") or {}).get("reporting_currency") or "USD"
        distress = distress_assessment(symbol, provider, financial, bundle, currency)
    except Exception as exc:  # the rest of the analysis is still useful
        distress = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    if distress["status"] == "available":
        rank = distress["riskier_than_percent_of_training"]
        drivers = ", ".join(f"{f['description'].lower()} ({f['effect']})"
                            for f in distress["top_factors"][:3])
        if distress["in_distress_now"]:
            signals.append(signal("distress_now", "high", "negative", "Currently in financial distress",
                                  f"Latest annual statements ({distress['fiscal_year_end']}) meet the distress definition."))
        if rank >= 95:
            severity = "high"
        elif rank >= 90:
            severity = "medium"
        elif rank >= 75:
            severity = "low"
        else:
            severity = None
        if severity:
            signals.append(signal("distress_model", severity, "negative",
                                  "Elevated distress-model score",
                                  f"Riskier than {rank}% of training company-years. Main drivers: {drivers}."))
        else:
            signals.append(signal("distress_model_low", "info", "positive",
                                  "Low distress-model score",
                                  f"Riskier than only {rank}% of training company-years. Main drivers: {drivers}."))

    flags = analysis["risk_flags"]["summary"]
    m = analysis["metrics"]["latest"] or {}
    ratios = m.get("ratios", {})
    parts = [f"{flags['triggered']} of {flags['evaluated']} rule-based flags triggered "
             f"({flags['not_evaluable']} not evaluable)."]
    if ratios.get("net_income_margin") is not None:
        parts.append(f"Latest quarter ({analysis['metrics']['latest_period']}): net margin "
                     f"{ratios['net_income_margin']:.1%}"
                     + (f", debt/equity {ratios['debt_to_equity']:.2f}" if ratios.get("debt_to_equity") is not None else "")
                     + (f", current ratio {ratios['current_ratio']:.2f}" if ratios.get("current_ratio") is not None else "")
                     + ".")
    if distress["status"] == "available":
        parts.append(f"Distress model: riskier than {distress['riskier_than_percent_of_training']}% "
                     f"of training company-years ({distress['fiscal_year_end']} statements).")
    else:
        parts.append(f"Distress model not applied: {distress['reason']}")

    data = {
        "latest_period": analysis["metrics"]["latest_period"],
        "ratios": ratios,
        "risk_flags_summary": flags,
        "distress": distress,
        "revenue_decline_signal": {
            "status": analysis["ml_signal"]["status"],
            "note": "Experimental; not better than chance in evaluation. Not used for signals.",
        },
        "data_quality": analysis["data_quality"]["finding_counts"],
    }
    status = "ok" if distress["status"] == "available" else "partial"
    return result("quantitative_risk", status, " ".join(parts), signals, data,
                  sources=[analysis["source"]["provider"], "SEC EDGAR (distress model training)"]), analysis
