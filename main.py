"""Multi-Agent Investment Research - command-line entry point.

Commands
  python main.py analyze RELIANCE.NS
  python main.py analyze "Reliance Industries"
      One company: resolve the ticker/name, collect quarterly statements,
      calculate ratios, audit data quality -> outputs/<SYMBOL>_<time>/

  python main.py rebuild
  python main.py rebuild --from-raw data/raw/quarterly
      Full quarterly pipeline (collect -> combine -> target -> features ->
      data-quality audit -> model scripts -> explanation -> evaluation) in a
      fresh runs/<time>/ folder. --from-raw works offline from stored
      statements.

  python main.py audit
      Data-quality audit of the committed datasets -> reports/

  python main.py evaluate
      Honest evaluation of the revenue-decline model (baselines, time- and
      company-aware validation, permutation test) -> reports/

  python main.py edgar
      SEC EDGAR quarterly history since ~2009 for the US companies
      (needs SEC_EDGAR_USER_AGENT in .env), cross-check with Yahoo, and the
      evaluation on next-quarter and year-over-year targets -> reports/sec_edgar/

  python main.py serve
      Local test page for `analyze` at http://127.0.0.1:8765

Run `python main.py <command> --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

EXIT_OK = 0
EXIT_FAILED_STEPS = 1
EXIT_BAD_INPUT = 2
EXIT_PROVIDER = 3


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_analyze(args):
    from src.pipeline.analyze import analyze_company

    summary = analyze_company(
        " ".join(args.company),
        output_root=args.output_dir,
        as_of=args.as_of,
        prefer_indian=args.prefer_indian,
    )
    if summary["status"] == "error":
        error = summary["error"]
        print(f"\nError ({error['code']}): {error['message']}")
        for candidate in error.get("candidates", [])[:8]:
            print(f"  - {candidate.get('symbol')}  {candidate.get('name') or ''}"
                  f"  {candidate.get('exchange') or ''}")
        print(f"Details: {summary['files']['summary']}")
        return EXIT_PROVIDER if error["code"] == "provider_error" else EXIT_BAD_INPUT

    company = summary["company"]
    coverage = summary["coverage"]
    metrics = summary["metrics"]
    print(f"\n{company['symbol']}: {company.get('name') or ''}")
    print(f"Exchange: {company.get('exchange')}; reporting currency: "
          f"{company.get('reporting_currency') or 'unknown'}")
    print(f"Quarters: {coverage['usable_quarters']} "
          f"({coverage['first_period']} to {coverage['latest_period']})")
    if metrics["latest"]:
        print(f"Latest quarter {metrics['latest_period']} ratios:")
        for key, value in metrics["latest"]["ratios"].items():
            shown = "n/a" if value is None else f"{value:.3f}"
            print(f"  {key:<18} {shown}")

    flags = summary["risk_flags"]
    print(f"Risk flags: {flags['summary']['triggered']} triggered, "
          f"{flags['summary']['not_evaluable']} not evaluable")
    for flag in flags["flags"]:
        if flag["status"] == "triggered":
            print(f"  [{flag['severity']}] {flag['title']}: {flag['explanation']}")

    signal = summary["ml_signal"]
    if signal["status"] == "available":
        print(f"ML signal (EXPERIMENTAL, not shown to beat chance): score "
              f"{signal['score']:.2f}, not a probability")
    else:
        print(f"ML signal: unavailable ({signal['reason']})")

    counts = summary["data_quality"]["finding_counts"]
    print(f"Data quality: {counts['error']} errors, {counts['warning']} "
          f"warnings, {counts['info']} notes")
    for warning in summary["warnings"]:
        print(f"  ! [{warning['code']}] {warning['message']}")
    print(f"\nReport: {summary['files']['report']}")
    print(summary["disclaimer"])
    return EXIT_OK


def cmd_rebuild(args):
    from src.pipeline.rebuild import DEFAULT_HISTORY_DIR, rebuild

    manifest = rebuild(
        run_dir=args.run_dir,
        tickers=args.tickers,
        from_raw=args.from_raw,
        history_dir=None if args.no_history else DEFAULT_HISTORY_DIR,
        as_of=args.as_of,
        permutations=args.permutations,
    )
    dataset = manifest.get("dataset")
    if dataset:
        print(f"\nLabelled rows: {dataset['labelled_rows']} from "
              f"{dataset['companies']} companies; labels {dataset['label_counts']}")
    print(f"Status: {manifest['status']}. Manifest: "
          f"{Path(manifest['run_dir']) / 'run_manifest.json'}")
    return EXIT_OK if manifest["status"] == "ok" else EXIT_FAILED_STEPS


def cmd_audit(args):
    from src.pipeline.quality import run_quality_checks

    processed = PROJECT_ROOT / "data" / "processed"
    try:
        results = run_quality_checks(
            processed / "quarterly_combined_dataset.csv",
            args.output_dir,
            raw_dir=PROJECT_ROOT / "data" / "raw" / "quarterly",
            ml_base_file=processed / "quarterly_ml_base_dataset.csv",
            target_file=processed / "quarterly_revenue_decline_target.csv",
            features_file=processed / "revenue_decline_features.csv",
            as_of=args.as_of,
            overwrite=args.overwrite,
            title="Data-quality audit: committed quarterly dataset",
        )
    except FileExistsError as exc:
        print(exc)
        return EXIT_BAD_INPUT
    counts = results["audit"]["finding_counts"]
    print(f"Findings: {counts['error']} errors, {counts['warning']} warnings, "
          f"{counts['info']} notes")
    ratio = results["ratio_check"]
    if ratio:
        print("Ratio formulas: " + json.dumps(
            {k: v["status"] for k, v in ratio["features"].items()}))
    if results["target_check"]:
        print("Target disagreements:",
              len(results["target_check"]["disagreements"]))
    print("Leakage problems:", results["feature_timing_problems"] or "none")
    print(f"Report: {results['files']['markdown']}")
    return EXIT_OK


def cmd_evaluate(args):
    from src.ml.evaluate_revenue_decline import evaluate

    processed = PROJECT_ROOT / "data" / "processed"
    try:
        results = evaluate(
            features_file=args.features or processed / "revenue_decline_features.csv",
            target_file=args.target or processed / "quarterly_revenue_decline_target.csv",
            output_dir=args.output_dir,
            n_permutations=args.permutations,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        print(exc)
        return EXIT_BAD_INPUT
    print()
    for line in results["conclusions"]:
        print(f"- {line}")
    print(f"\nReport: {results['files']['markdown']}")
    return EXIT_OK


def cmd_edgar(args):
    from src.data_collection.sec_edgar import EdgarConfigError
    from src.data_collection.provider import ProviderError
    from src.pipeline.edgar import run_edgar

    try:
        summary, results = run_edgar(
            tickers=args.tickers, skip_download=args.skip_download,
            permutations=args.permutations, overwrite=args.overwrite,
        )
    except (FileExistsError, EdgarConfigError) as exc:
        print(exc)
        return EXIT_BAD_INPUT
    except ProviderError as exc:
        print(f"SEC EDGAR download failed: {exc}")
        return EXIT_PROVIDER
    check = summary["cross_check"]["fields"]
    print("\nCross-check with Yahoo: " + ", ".join(
        f"{k} {v['matching']}/{v['compared']}" for k, v in check.items()))
    for name, result in results.items():
        print(f"\n[{name}] {result['target']['name']}")
        for line in result["conclusions"]:
            print(f"- {line}")
    print("\nReports: reports/sec_edgar/")
    return EXIT_OK


def cmd_serve(args):
    from src.web.server import serve

    serve(host=args.host, port=args.port, output_root=args.output_dir)
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(
        description="Investment research pipeline (informational use only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyse one company")
    analyze.add_argument("company", nargs="+",
                         help="ticker (RELIANCE.NS, AAPL) or company name")
    analyze.add_argument("--output-dir", type=Path, default=None,
                         help="parent folder for results (default: outputs/)")
    analyze.add_argument("--prefer-indian", action="store_true",
                         help="for a bare symbol, try NSE/BSE before the "
                              "symbol as typed")
    analyze.add_argument("--as-of", default=None,
                         help="date for the stale-data check (YYYY-MM-DD)")
    analyze.set_defaults(func=cmd_analyze)

    rebuild = sub.add_parser("rebuild", help="rerun the full quarterly pipeline")
    rebuild.add_argument("--tickers", nargs="+", default=None,
                         help="companies to include (default: the project list)")
    rebuild.add_argument("--from-raw", type=Path, default=None,
                         help="copy raw statements from this folder instead of "
                              "downloading (offline)")
    rebuild.add_argument("--no-history", action="store_true",
                         help="do not merge downloads with data/raw/quarterly")
    rebuild.add_argument("--run-dir", type=Path, default=None,
                         help="output folder (default: runs/<timestamp>/)")
    rebuild.add_argument("--as-of", default=None,
                         help="date for the stale-data check (YYYY-MM-DD)")
    rebuild.add_argument("--permutations", type=int, default=200,
                         help="label shuffles in the evaluation step (0 to skip)")
    rebuild.set_defaults(func=cmd_rebuild)

    audit = sub.add_parser("audit", help="audit the committed datasets")
    audit.add_argument("--output-dir", type=Path,
                       default=PROJECT_ROOT / "reports")
    audit.add_argument("--overwrite", action="store_true",
                       help="replace an existing audit report")
    audit.add_argument("--as-of", default=None,
                       help="date for the stale-data check (YYYY-MM-DD)")
    audit.set_defaults(func=cmd_audit)

    evaluate = sub.add_parser(
        "evaluate", help="evaluate the revenue-decline model honestly")
    evaluate.add_argument("--features", type=Path, default=None,
                          help="feature dataset (default: committed data)")
    evaluate.add_argument("--target", type=Path, default=None,
                          help="target dataset with Next_Date (default: committed)")
    evaluate.add_argument("--output-dir", type=Path,
                          default=PROJECT_ROOT / "reports")
    evaluate.add_argument("--permutations", type=int, default=200,
                          help="label shuffles for the p-value (0 to skip)")
    evaluate.add_argument("--overwrite", action="store_true",
                          help="replace an existing evaluation report")
    evaluate.set_defaults(func=cmd_evaluate)

    edgar = sub.add_parser(
        "edgar", help="long history from SEC EDGAR (US companies) + evaluation")
    edgar.add_argument("--tickers", nargs="+", default=None,
                       help="US tickers (default: AAPL AMZN GOOGL META MSFT NVDA TSLA)")
    edgar.add_argument("--skip-download", action="store_true",
                       help="reuse data/raw/sec_edgar/quarterly instead of downloading")
    edgar.add_argument("--permutations", type=int, default=200,
                       help="label shuffles for the p-values (0 to skip)")
    edgar.add_argument("--overwrite", action="store_true",
                       help="replace existing EDGAR evaluation reports")
    edgar.set_defaults(func=cmd_edgar)

    serve = sub.add_parser("serve", help="local test page for analyze")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--host", default="127.0.0.1",
                       help="interface to listen on (default: local only)")
    serve.add_argument("--output-dir", type=Path, default=None,
                       help="parent folder for results (default: outputs/)")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv=None):
    _utf8_console()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
