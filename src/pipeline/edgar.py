"""SEC EDGAR pipeline: download -> cross-check -> dataset -> evaluation.

    python main.py edgar                  # download (cached 7 days) and evaluate
    python main.py edgar --skip-download  # reuse data/raw/sec_edgar/quarterly

Outputs:
  data/raw/sec_edgar/quarterly/<TICKER>_quarterly.csv   point-in-time quarters
  data/processed/sec_edgar_revenue_decline_features.csv labelled dataset
  reports/sec_edgar/collection.md / .json               coverage + cross-check
  reports/sec_edgar/qoq/model_evaluation.*              next-quarter target
  reports/sec_edgar/yoy/model_evaluation.*              year-over-year target
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_collection import edgar_quarterly
from src.ml import edgar_dataset
from src.ml.evaluate_revenue_decline import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "sec_edgar"
YAHOO_COMBINED = PROJECT_ROOT / "data" / "processed" / "quarterly_combined_dataset.csv"


def _collection_markdown(summary):
    out = [
        "# SEC EDGAR quarterly history",
        "",
        f"Generated {summary['generated_at']}. Source: SEC EDGAR XBRL company "
        "facts (10-Q/10-K), first-filed values (as originally reported).",
        "",
        "| Company | Quarters | Point-in-time | Period range | Note |",
        "|---|---|---|---|---|",
    ]
    for ticker, info in summary["companies"].items():
        out.append(
            f"| {ticker} | {info.get('quarters', 0)} | {info.get('point_in_time', 0)} "
            f"| {info.get('first', '')} to {info.get('last', '')} | {info['note']} |"
        )
    check = summary["cross_check"]
    out += [
        "",
        "Point-in-time = revenue first filed within "
        f"{edgar_quarterly.MAX_FILING_LAG_DAYS} days of the quarter end. "
        "Other quarters (mostly before XBRL tagging began, 2009-2011) appear "
        "only as comparatives in later, possibly restated filings and are "
        "excluded from modelling.",
        "",
        "## Cross-check with Yahoo Finance",
        "",
        f"Same company and quarter (period ends within 7 days); values agree "
        f"if within {check['tolerance']:.1%}.",
        "",
        "| Field | Matching / compared |",
        "|---|---|",
    ]
    for field, s in check["fields"].items():
        note = " (different definition: EDGAR excludes leases)" \
            if field == "Total Debt" else ""
        out.append(f"| {field} | {s['matching']}/{s['compared']}{note} |")
    if check["mismatches"]:
        out += ["", "Mismatches:", ""]
        out += [
            f"- {m['ticker']} {m['period']} {m['field']}: Yahoo {m['yahoo']:,.0f}, "
            f"EDGAR {m['edgar']:,.0f}"
            for m in check["mismatches"]
        ]
    out.append("")
    return "\n".join(out)


def run_edgar(tickers=None, skip_download=False, permutations=200,
              overwrite=False, report_dir=REPORT_DIR, echo=print):
    report_dir = Path(report_dir)
    targets = {
        "qoq": "Next_Period_Revenue_Decline",
        "yoy": "Next_Period_Revenue_Decline_YoY",
    }
    if not overwrite:
        existing = [report_dir / name / "model_evaluation.md" for name in targets]
        if any(p.exists() for p in existing):
            raise FileExistsError(
                f"EDGAR reports already exist in {report_dir}; use --overwrite."
            )

    tickers = tickers or edgar_quarterly.DEFAULT_TICKERS
    if skip_download:
        tables = {}
        for ticker in tickers:
            path = edgar_quarterly.OUTPUT_DIR / f"{ticker}_quarterly.csv"
            if path.exists():
                tables[ticker] = pd.read_csv(path)
        notes = {t: "loaded from disk" for t in tables}
        echo(f"Loaded {len(tables)} companies from {edgar_quarterly.OUTPUT_DIR}")
    else:
        from src.data_collection.sec_edgar import EdgarClient

        echo("Downloading SEC EDGAR company facts...")
        tables, notes = edgar_quarterly.collect(EdgarClient(), tickers, echo=echo)
    if not tables:
        raise RuntimeError("No EDGAR quarterly data available.")

    companies = {}
    for ticker in tickers:
        info = {"note": notes.get(ticker, "no data")}
        if ticker in tables:
            table = tables[ticker]
            pit = table["Point_In_Time"].astype(str).str.lower() == "true"
            info.update({
                "quarters": int(len(table)),
                "point_in_time": int(pit.sum()),
                "first": str(table["Date"].min())[:10],
                "last": str(table["Date"].max())[:10],
            })
        companies[ticker] = info
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "companies": companies,
        "cross_check": edgar_quarterly.cross_check(tables, YAHOO_COMBINED),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "collection.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (report_dir / "collection.md").write_text(
        _collection_markdown(summary), encoding="utf-8")

    echo("Building the labelled dataset...")
    data = edgar_dataset.build_dataset(edgar_quarterly.OUTPUT_DIR, tickers)
    edgar_dataset.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(edgar_dataset.OUTPUT_FILE, index=False)
    echo(f"  {len(data)} labelled quarters from {data['Ticker'].nunique()} companies")

    results = {}
    for name, column in targets.items():
        echo(f"Evaluating target {column}...")
        results[name] = evaluate(
            features_file=edgar_dataset.OUTPUT_FILE,
            output_dir=report_dir / name,
            n_permutations=permutations,
            overwrite=overwrite,
            target_column=column,
            dataset_label=(
                f"SEC EDGAR point-in-time quarters, {len(data)} rows, "
                f"{data['Ticker'].nunique()} US companies"
            ),
            echo=echo,
        )
    return summary, results
