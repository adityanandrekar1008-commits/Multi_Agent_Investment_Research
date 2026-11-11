"""Run the data-quality audit and feature checks on pipeline files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_quality.audit import audit_quarterly
from src.data_quality.report import render_audit_markdown
from src.data_quality.verify_features import (
    check_feature_timing,
    verify_ratio_formulas,
    verify_target,
)


def model_feature_lists():
    """Collect the feature lists used by every ML script."""
    from src.explainability import explain_revenue_decline
    from src.ml import (
        compare_company_feature_sets,
        compare_feature_sets,
        inspect_prediction_errors,
        train_revenue_decline_model,
        validate_company_wise,
        validate_date_grouped,
    )

    lists = {}
    for module in [
        train_revenue_decline_model,
        validate_date_grouped,
        validate_company_wise,
        inspect_prediction_errors,
        explain_revenue_decline,
    ]:
        lists[module.__name__.rsplit(".", 1)[-1]] = list(module.FEATURES)
    for module in [compare_feature_sets, compare_company_feature_sets]:
        for set_name, features in module.FEATURE_SETS.items():
            name = f"{module.__name__.rsplit('.', 1)[-1]}[{set_name}]"
            lists[name] = list(features)
    return lists


def run_quality_checks(
    combined_file,
    output_dir,
    raw_dir=None,
    ml_base_file=None,
    target_file=None,
    features_file=None,
    as_of=None,
    currencies=None,
    overwrite=False,
    title="Data-quality audit",
):
    """Audit the combined quarterly dataset and verify features/target.

    Writes data_quality_audit.md and data_quality_audit.json to output_dir
    and returns the results dict.
    """
    output_dir = Path(output_dir)
    md_file = output_dir / "data_quality_audit.md"
    json_file = output_dir / "data_quality_audit.json"
    if not overwrite and (md_file.exists() or json_file.exists()):
        raise FileExistsError(
            f"{md_file.name} already exists in {output_dir}; pass "
            "overwrite=True (--overwrite) to replace it."
        )

    combined = pd.read_csv(combined_file)
    audit = audit_quarterly(
        combined, as_of=as_of, currencies=currencies, raw_dir=raw_dir
    )

    ratio_check = None
    if features_file and Path(features_file).exists():
        ratio_check = verify_ratio_formulas(pd.read_csv(features_file))

    target_check = None
    if (
        ml_base_file and target_file
        and Path(ml_base_file).exists() and Path(target_file).exists()
    ):
        target_check = verify_target(
            pd.read_csv(ml_base_file), pd.read_csv(target_file)
        )

    timing_problems = check_feature_timing(model_feature_lists())

    results = {
        "audit": audit,
        "ratio_check": ratio_check,
        "feature_timing_problems": timing_problems,
        "target_check": target_check,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    md_file.write_text(
        render_audit_markdown(
            audit, ratio_check, timing_problems, target_check, title=title
        ),
        encoding="utf-8",
    )
    json_file.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    results["files"] = {"markdown": str(md_file), "json": str(json_file)}
    return results
