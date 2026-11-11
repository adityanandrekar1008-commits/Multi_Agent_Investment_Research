"""Readable Markdown report generated from a company_analysis/1.0 payload.

The report only presents what is in the JSON, so the app and the report
can never disagree.
"""

from __future__ import annotations


def _money(value, currency):
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= limit:
            return f"{sign}{value / limit:.2f}{suffix} {currency or ''}".strip()
    return f"{sign}{value:.0f} {currency or ''}".strip()


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _times(value):
    return "n/a" if value is None else f"{value:.2f}x"


def _cell(text):
    return str(text).replace("|", "/").replace("\n", " ")


def render_company_report(payload):
    out = []
    if payload["status"] == "error":
        error = payload["error"]
        out += [
            f"# Company analysis: {payload['request']['input']}",
            "",
            f"**Error `{error['code']}`:** {error['message']}",
            "",
        ]
        if error.get("candidates"):
            out += ["Matches found:", ""]
            out += [
                f"- {c.get('symbol')} {c.get('name') or ''} "
                f"{('(' + c['exchange'] + ')') if c.get('exchange') else ''}".rstrip()
                for c in error["candidates"]
            ]
            out.append("")
        out += [f"_{payload['disclaimer']}_", ""]
        return "\n".join(out)

    company = payload["company"]
    source = payload["source"]
    coverage = payload["coverage"]
    metrics = payload["metrics"]
    currency = metrics.get("currency")
    latest = metrics.get("latest") or {}
    values = latest.get("values", {})
    ratios = latest.get("ratios", {})
    changes = latest.get("changes", {})

    out += [
        f"# {company.get('name') or company['symbol']} ({company['symbol']})",
        "",
        f"_{payload['disclaimer']}_",
        "",
        "## Source and coverage",
        "",
        f"- Exchange: {company.get('exchange') or 'n/a'}; sector: "
        f"{company.get('sector') or 'n/a'}; country: {company.get('country') or 'n/a'}",
        f"- Data: {source['provider']}, retrieved {source['retrieved_at']} (UTC)",
        f"- Reporting currency: {currency or 'unknown'}",
        f"- Quarters: {coverage['usable_quarters']} "
        f"({coverage['first_period']} to {coverage['latest_period']})",
        "- Statements: " + ", ".join(
            f"{name.replace('_', ' ')} "
            + (f"{s['periods']} periods" if s["available"] else "not available")
            for name, s in source["statements"].items()
        ),
        "",
        f"## Key figures, quarter ended {metrics.get('latest_period')}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Revenue | {_money(values.get('revenue'), currency)} |",
        f"| Revenue vs previous quarter | {_pct(changes.get('revenue_change_qoq'))} |",
        f"| Revenue vs same quarter last year | {_pct(changes.get('revenue_change_yoy'))} |",
        f"| Net income | {_money(values.get('net_income'), currency)} |",
        f"| Operating income | {_money(values.get('operating_income'), currency)} |",
        f"| Net income margin | {_pct(ratios.get('net_income_margin'))} |",
        f"| Operating margin | {_pct(ratios.get('operating_margin'))} |",
        f"| Debt / assets | {_pct(ratios.get('debt_to_assets'))} |",
        f"| Debt / equity | {_times(ratios.get('debt_to_equity'))} |",
        f"| Current ratio | {_times(ratios.get('current_ratio'))} |",
        f"| Operating cash flow | {_money(values.get('operating_cash_flow'), currency)} |",
        f"| Free cash flow | {_money(values.get('free_cash_flow'), currency)} |",
        "",
        "n/a = not reported for this quarter.",
        "",
    ]

    flags = payload["risk_flags"]
    summary = flags["summary"]
    by_severity = summary["triggered_by_severity"]
    out += [
        "## Risk flags (rule-based)",
        "",
        f"{summary['triggered']} of {summary['evaluated']} evaluable rules "
        f"triggered (high {by_severity['high']}, medium {by_severity['medium']}, "
        f"low {by_severity['low']}); {summary['not_evaluable']} could not be "
        "evaluated. These are general screening rules with fixed thresholds, "
        "not a rating.",
        "",
        "| Flag | Status | Severity | Value | Threshold | Period | Explanation |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {"triggered": 0, "not_triggered": 1, "not_evaluable": 2}
    for flag in sorted(flags["flags"], key=lambda f: order[f["status"]]):
        value = flag["value"]
        shown = "" if value is None else (
            f"{value:.3g}" if abs(value) < 1e6 else _money(value, currency)
        )
        out.append(
            f"| {_cell(flag['title'])} | {flag['status'].replace('_', ' ')} | "
            f"{flag['severity']} | {shown} | "
            f"{'' if flag['threshold'] is None else flag['threshold']} | "
            f"{flag['period_end'] or ''} | {_cell(flag['explanation'])} |"
        )
    out.append("")

    signal = payload["ml_signal"]
    out += ["## Experimental ML signal", ""]
    if signal["status"] != "available":
        out += [f"Not available: {signal['reason']}", ""]
    else:
        out += [
            f"**Experimental; research use only.** Model `{signal['model']}`: "
            f"{signal['target']} (horizon: {signal['horizon']}).",
            "",
            f"- Score for {signal['period_end']}: {signal['score']:.2f} "
            f"({signal['predicted_class'].replace('_', ' ')}). "
            f"{signal['score_meaning']}",
            "- Main contributions: " + "; ".join(
                f"{c['feature']} ({c['direction']})"
                for c in signal["top_contributions"]
            ) + ".",
        ]
        imputed = [f for f, i in signal["inputs"].items() if i["imputed"]]
        if imputed:
            out.append(
                "- Missing inputs replaced by the training median: "
                + ", ".join(imputed) + "."
            )
        out.append("")
    evaluation = signal.get("evaluation", {})
    if evaluation.get("verdict"):
        out += [f"Evaluation: {evaluation['verdict']}", ""]
    out += ["Limitations:", ""]
    out += [f"- {item}" for item in signal["limitations"]]
    out.append("")

    out += ["## Warnings", ""]
    if payload["warnings"]:
        out += [f"- `{w['code']}`: {w['message']}" for w in payload["warnings"]]
    else:
        out.append("None.")
    counts = payload["data_quality"]["finding_counts"]
    out += [
        "",
        f"Data-quality audit: {counts['error']} errors, {counts['warning']} "
        f"warnings, {counts['info']} notes (details: "
        f"{payload['files'].get('data_quality_report', 'data_quality.md')}).",
        "",
        "## Ratio definitions",
        "",
    ]
    out += [f"- {name}: {text}" for name, text in metrics["ratio_definitions"].items()]
    out.append("")
    return "\n".join(out)
