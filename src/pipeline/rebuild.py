"""Rebuild the quarterly dataset and rerun every ML step in a fresh folder.

Each run writes to its own folder (runs/<timestamp>/ by default), so the
committed data/ and reports/ files are never overwritten and the scripts'
"output already exists" guards never block a rerun.

Steps (each step's console output goes to logs/NN_<step>.log):
 1. collect       download quarterly statements (or copy them with --from-raw)
                  and merge them with stored history so older quarters are kept
 2. combine       src/data_collection/prepare_quarterly_dataset.py
 3. ml_base       src/data_collection/prepare_quarterly_ml_data.py
 4. target        src/data_collection/create_revenue_decline_target.py
 5. features      src/ml/prepare_revenue_decline_features.py
 6. data_quality  audit + ratio/target/leakage checks (src/data_quality)
 7-12. evaluation train, date-grouped and company-wise validation, feature-set
                  comparisons, prediction-error inspection
13. explain       src/explainability/explain_revenue_decline.py
14. evaluate      src/ml/evaluate_revenue_decline.py (baselines, time- and
                  company-aware validation, permutation test)

The existing scripts are reused unchanged: their module-level path
constants are pointed at the run folder for the duration of each step.
"""

from __future__ import annotations

import contextlib
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "data" / "raw" / "quarterly"
STATEMENTS = ["income_statement", "balance_sheet", "cash_flow"]
EXPECTED_STEPS = 14


@contextlib.contextmanager
def patched(module, **attributes):
    """Temporarily replace module-level constants (paths, ticker lists)."""
    original = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def merge_raw_history(fresh_dir, history_dir, output_dir, tickers):
    """Merge freshly downloaded statements with stored ones.

    Yahoo Finance serves only the latest ~5-7 quarters, and its oldest
    column is often nearly empty (e.g. 5 of 34 income fields), so a plain
    re-download loses data. For each company/statement:
    - every period from either source is kept;
    - for a period in both, each field takes the fresh value when the fresh
      download has one, otherwise the stored value.
    Returns {file name: {"fresh": n, "history_only": n, "fields_filled": n}}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    def load(path):
        frame = pd.read_csv(path)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"])
        return frame.drop_duplicates("Date", keep="last").set_index("Date")

    for ticker in tickers:
        for statement in STATEMENTS:
            name = f"{ticker}_{statement}.csv"
            fresh_path = Path(fresh_dir) / name
            old_path = Path(history_dir) / name if history_dir else None
            fresh = load(fresh_path) if fresh_path.exists() else None
            old = load(old_path) if old_path is not None and old_path.exists() else None
            if fresh is None and old is None:
                continue

            if fresh is None:
                merged, history_only, filled = old, len(old), 0
            elif old is None:
                merged, history_only, filled = fresh, 0, 0
            else:
                merged = fresh.combine_first(old)
                history_only = len(old.index.difference(fresh.index))
                overlap = fresh.index.intersection(old.index)
                common = fresh.columns.intersection(old.columns)
                filled = int((
                    fresh.loc[overlap, common].isna()
                    & old.loc[overlap, common].notna()
                ).to_numpy().sum())

            merged = merged.sort_index(ascending=False).reset_index()
            merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
            merged["Ticker"] = ticker
            # Keep Date and Ticker first, as in the collector's files.
            columns = ["Date", "Ticker"] + [
                c for c in merged.columns if c not in ("Date", "Ticker")
            ]
            merged[columns].to_csv(output_dir / name, index=False)
            summary[name] = {
                "fresh": 0 if fresh is None else len(fresh),
                "history_only": int(history_only),
                "fields_filled": filled,
            }
    return summary


def _package_versions():
    versions = {"python": platform.python_version()}
    for package in ["pandas", "numpy", "scikit-learn", "yfinance"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


class _Run:
    def __init__(self, run_dir, echo):
        self.run_dir = run_dir
        self.log_dir = run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.echo = echo

    def step(self, name, func, outputs=(), required=True):
        index = len(self.steps) + 1
        log_file = self.log_dir / f"{index:02d}_{name}.log"
        start = time.time()
        status, message, detail = "ok", "", None
        with open(log_file, "w", encoding="utf-8") as log:
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    detail = func()
            except Exception as exc:  # record and continue/stop below
                status = "failed"
                message = f"{type(exc).__name__}: {exc}"
                log.write("\n" + traceback.format_exc())
        missing = [str(p) for p in outputs if not Path(p).exists()]
        if status == "ok" and missing:
            status = "failed"
            message = "Expected output not created: " + ", ".join(
                Path(p).name for p in missing
            ) + " (see log)"
        record = {
            "step": name,
            "status": status,
            "seconds": round(time.time() - start, 1),
            "log": str(log_file.relative_to(self.run_dir)),
            "outputs": [str(Path(p).relative_to(self.run_dir)) for p in outputs],
            "message": message,
        }
        if isinstance(detail, dict):
            record["detail"] = detail
        self.steps.append(record)
        self.echo(f"  [{status:6}] {record['seconds']:>5}s  {name}"
                  + (f"  - {message}" if message else ""))
        if status != "ok" and required:
            raise _StopRun(name)
        return status == "ok"


class _StopRun(Exception):
    pass


def rebuild(run_dir=None, tickers=None, from_raw=None,
            history_dir=DEFAULT_HISTORY_DIR, as_of=None, permutations=200,
            echo=print):
    """Run the full quarterly pipeline into run_dir. Returns the manifest."""
    from src.data_collection import (
        collect_quarterly_data as collect,
        create_revenue_decline_target as target_script,
        prepare_quarterly_dataset as combine_script,
        prepare_quarterly_ml_data as base_script,
    )
    from src.explainability import explain_revenue_decline as explain
    from src.ml import (
        compare_company_feature_sets,
        compare_feature_sets,
        evaluate_revenue_decline as evaluator,
        inspect_prediction_errors,
        prepare_revenue_decline_features as features_script,
        train_revenue_decline_model as train,
        validate_company_wise,
        validate_date_grouped,
    )
    from src.data_collection.provider import YahooFinanceProvider
    from src.pipeline.quality import run_quality_checks

    run_dir = Path(
        run_dir or PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run folder is not empty: {run_dir}")

    tickers = list(tickers or collect.TICKERS)
    raw_dir = run_dir / "data" / "raw" / "quarterly"
    downloaded_dir = run_dir / "data" / "raw" / "downloaded"
    processed = run_dir / "data" / "processed"
    reports = run_dir / "reports"
    for folder in (raw_dir, processed, reports):
        folder.mkdir(parents=True, exist_ok=True)

    files = {
        "combined": processed / "quarterly_combined_dataset.csv",
        "base": processed / "quarterly_ml_base_dataset.csv",
        "target": processed / "quarterly_revenue_decline_target.csv",
        "features": processed / "revenue_decline_features.csv",
    }

    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "tickers": tickers,
        "source": (
            f"copied from {from_raw}" if from_raw
            else "downloaded from Yahoo Finance"
        ),
        "history_dir": None if (from_raw or not history_dir) else str(history_dir),
        "versions": _package_versions(),
        "command": " ".join(sys.argv),
    }
    echo(f"Run folder: {run_dir}")
    run = _Run(run_dir, echo)
    currencies = {}

    def collect_step():
        if from_raw:
            copied = []
            for ticker in tickers:
                for statement in STATEMENTS:
                    source = Path(from_raw) / f"{ticker}_{statement}.csv"
                    if source.exists():
                        shutil.copy2(source, raw_dir / source.name)
                        copied.append(source.name)
            print("Copied:", copied)
            detail = {"copied_files": len(copied)}
        else:
            downloaded_dir.mkdir(parents=True, exist_ok=True)
            with patched(collect, OUTPUT_DIR=downloaded_dir, TICKERS=tickers):
                collect.main()
            merge = merge_raw_history(
                downloaded_dir, history_dir, raw_dir, tickers
            )
            print("Merged with history:", json.dumps(merge, indent=2))
            provider = YahooFinanceProvider()
            for ticker in tickers:
                currency = provider.profile(ticker).get("financialCurrency")
                if currency:
                    currencies[ticker] = currency
            detail = {"files": merge, "reporting_currencies": currencies}

        income = sorted(raw_dir.glob("*_income_statement.csv"))
        if not income:
            raise RuntimeError("No income statements were collected.")
        collected = {
            t: [s for s in STATEMENTS if (raw_dir / f"{t}_{s}.csv").exists()]
            for t in tickers
        }
        detail["statements_by_company"] = collected
        return detail

    try:
        run.step("collect", collect_step)

        def combine_step():
            with patched(combine_script, INPUT_DIR=raw_dir, OUTPUT_DIR=processed,
                         OUTPUT_FILE=files["combined"]):
                combine_script.main()
        run.step("combine", combine_step, [files["combined"]])

        def base_step():
            with patched(base_script, INPUT_FILE=files["combined"],
                         OUTPUT_FILE=files["base"]):
                base_script.main()
        run.step("ml_base", base_step, [files["base"]])

        def target_step():
            with patched(target_script, INPUT_FILE=files["base"],
                         OUTPUT_FILE=files["target"]):
                target_script.main()
        run.step("target", target_step, [files["target"]])

        def features_step():
            with patched(features_script, INPUT_FILE=files["target"],
                         OUTPUT_FILE=files["features"]):
                features_script.main()
        run.step("features", features_step, [files["features"]])

        def quality_step():
            results = run_quality_checks(
                files["combined"], reports, raw_dir=raw_dir,
                ml_base_file=files["base"], target_file=files["target"],
                features_file=files["features"], as_of=as_of,
                currencies=currencies,
            )
            return {"finding_counts": results["audit"]["finding_counts"]}
        run.step("data_quality", quality_step,
                 [reports / "data_quality_audit.md"], required=False)

        evaluation = [
            ("train_baseline", train, "revenue_decline_baseline_results.txt"),
            ("validate_date_grouped", validate_date_grouped,
             "date_grouped_validation_results.txt"),
            ("validate_company_wise", validate_company_wise,
             "company_wise_validation_results.csv"),
            ("compare_feature_sets", compare_feature_sets,
             "feature_set_comparison.csv"),
            ("compare_company_feature_sets", compare_company_feature_sets,
             "company_feature_set_comparison.csv"),
            ("inspect_prediction_errors", inspect_prediction_errors,
             "prediction_error_inspection.csv"),
        ]
        for name, module, output in evaluation:
            def eval_step(module=module, output=output):
                with patched(module, INPUT_FILE=files["features"],
                             OUTPUT_FILE=reports / output):
                    module.main()
            run.step(name, eval_step, [reports / output], required=False)

        def explain_step():
            with patched(
                explain, INPUT_FILE=files["features"], REPORT_DIR=reports,
                PREDICTIONS_FILE=reports / "revenue_decline_predictions.csv",
                EXPLANATIONS_FILE=reports / "revenue_decline_feature_explanations.csv",
            ):
                explain.main()
        run.step("explain", explain_step, [
            reports / "revenue_decline_predictions.csv",
            reports / "revenue_decline_feature_explanations.csv",
        ], required=False)

        def evaluate_step():
            results = evaluator.evaluate(
                files["features"], files["target"], reports,
                n_permutations=permutations,
            )
            return {"conclusions": results["conclusions"]}
        run.step("evaluate", evaluate_step,
                 [reports / "model_evaluation.md"], required=False)

    except _StopRun as stop:
        echo(f"Stopped: required step '{stop}' failed. See its log.")

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["steps"] = run.steps
    manifest["status"] = (
        "ok" if all(s["status"] == "ok" for s in run.steps)
        and len(run.steps) == EXPECTED_STEPS else "failed"
    )

    if files["target"].exists():
        labelled = pd.read_csv(files["target"])
        manifest["dataset"] = {
            "labelled_rows": int(len(labelled)),
            "companies": int(labelled["Ticker"].nunique()),
            "label_counts": {
                str(k): int(v) for k, v in
                labelled["Next_Period_Revenue_Decline"].value_counts().sort_index().items()
            },
            "date_range": [
                str(labelled["Date"].min())[:10], str(labelled["Date"].max())[:10]
            ],
        }

    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest
