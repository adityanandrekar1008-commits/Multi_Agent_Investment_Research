"""Render data-quality and feature-check results as Markdown."""

from __future__ import annotations


def _pct(value):
    return f"{value:.0%}"


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_audit_markdown(audit, ratio_check=None, timing_problems=None,
                          target_check=None, title="Data-quality audit"):
    counts = audit["finding_counts"]
    out = [
        f"# {title}",
        "",
        f"- As of: {audit['as_of']}",
        f"- Companies: {audit['companies_audited']}; records: {audit['records']}",
        f"- Findings: {counts['error']} errors, {counts['warning']} warnings, "
        f"{counts['info']} notes",
        "",
        "The audit only reads data; no values were changed, filled, or "
        "removed. Findings flag things to review; they are not proof of "
        "errors.",
        "",
        "## Coverage by company",
        "",
    ]

    coverage_fields = [
        "Total Revenue", "Net Income", "Operating Income", "Total Assets",
        "Total Debt", "Stockholders Equity", "Current Assets",
        "Operating Cash Flow",
    ]
    short = {
        "Total Revenue": "Revenue", "Net Income": "Net inc.",
        "Operating Income": "Op. inc.", "Total Assets": "Assets",
        "Total Debt": "Debt", "Stockholders Equity": "Equity",
        "Current Assets": "Cur. assets", "Operating Cash Flow": "Op. CF",
    }
    rows = []
    for ticker, info in audit["companies"].items():
        currency = info.get("reporting_currency", {})
        rows.append(
            [
                ticker,
                info["usable_quarters"],
                f"{info['first_period']} to {info['latest_period']}",
                currency.get("currency", "?")
                + ("*" if currency.get("source", "").startswith("inferred") else ""),
            ]
            + [_pct(info["key_field_coverage"].get(f, 0)) for f in coverage_fields]
        )
    out.append(_table(
        ["Ticker", "Quarters", "Periods", "Currency"]
        + [short[f] for f in coverage_fields],
        rows,
    ))
    out += [
        "",
        "Coverage = share of usable quarters with a value. "
        "\\* currency inferred from the ticker suffix, not reported by the "
        "provider.",
        "",
    ]

    out += ["## Findings", ""]
    if not audit["findings"]:
        out.append("No findings.")
    else:
        rows = [
            [
                f["severity"],
                f["ticker"] or "all",
                f["check"],
                f["period"] or "",
                f["message"].replace("|", "/"),
            ]
            for f in audit["findings"]
        ]
        out.append(_table(["Severity", "Ticker", "Check", "Period", "Detail"], rows))
    out.append("")

    if ratio_check is not None:
        out += [
            "## Ratio formula check",
            "",
            "Each stored ratio was recomputed with plain arithmetic, "
            "independently of the pipeline code.",
            "",
        ]
        rows = []
        for feature, result in ratio_check["features"].items():
            rows.append([
                feature,
                result.get("formula", ""),
                result["status"],
                f"{result.get('rows_matching', '-')}/{result.get('rows_checked', '-')}",
                result.get("missing_values", "-"),
            ])
        out.append(_table(
            ["Feature", "Formula", "Status", "Matching rows", "Missing"], rows
        ))
        if ratio_check["mismatches"]:
            out += ["", "Mismatches:", ""]
            out.append(_table(
                ["Feature", "Ticker", "Date", "Stored", "Expected"],
                [[m["feature"], m["ticker"], m["date"], m["stored"], m["expected"]]
                 for m in ratio_check["mismatches"][:50]],
            ))
        out.append("")

    if timing_problems is not None:
        out += ["## Feature timing (leakage) check", ""]
        if timing_problems:
            out += [f"- {p}" for p in timing_problems]
        else:
            out.append(
                "- No model feature is a future/outcome column or an "
                "identifier, and every feature is a documented "
                "current-period ratio."
            )
        out += [
            "- Note: `Date` is the fiscal period end. Statements are "
            "published weeks later (typically 25-60 days), so a prediction "
            "for the next quarter can only be made after the filing date. "
            "The provider does not give filing dates, so this is not checked "
            "per row.",
            "",
        ]

    if target_check is not None:
        out += [
            "## Target check",
            "",
            f"- Definition: {target_check['definition']}",
            f"- Horizon: {target_check['horizon']}",
            f"- Labels recomputed: {target_check['expected_labels']}; "
            f"stored: {target_check['stored_labels']}; "
            f"disagreements: {len(target_check['disagreements'])}",
        ]
        for d in target_check["disagreements"][:20]:
            out.append(
                f"  - {d['ticker']} {d['date']}: expected {d['expected']}, "
                f"stored {d['stored']}"
            )
        out.append("")

    return "\n".join(out)
