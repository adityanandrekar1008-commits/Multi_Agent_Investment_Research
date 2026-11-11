
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow `python src/ml/train_revenue_decline_model.py` as well as imports.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.splits import date_boundary_split  # noqa: E402

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "revenue_decline_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "revenue_decline_baseline_results.txt"
)

TARGET = "Next_Period_Revenue_Decline"

FEATURES = [
    "Net_Income_Margin",
    "Operating_Margin",
    "Debt_To_Assets",
    "Debt_To_Equity",
    "Current_Ratio",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET])
    df = df.sort_values("Date").reset_index(drop=True)

    available_features = [
        column for column in FEATURES
        if column in df.columns
    ]

    if len(available_features) < 1:
        raise ValueError("No usable model features were found.")

    # Keep the later dates as the test set. Split on a date boundary so no
    # reporting date is in both sets (a row-position split put some
    # companies' 2025-12-31 rows in training and others in test).
    train, test, _ = date_boundary_split(df, train_share=0.70)

    X_train = train[available_features]
    y_train = train[TARGET].astype(int)

    X_test = test[available_features]
    y_test = test[TARGET].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one target class. "
            "A classifier cannot learn both classes from this split."
        )

    # All imputation and scaling are fitted on training data only.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    # Simple baseline: always predict the most common training class.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    model.fit(X_train, y_train)

    model_predictions = model.predict(X_test)
    dummy_predictions = dummy.predict(X_test)

    results = []

    results.append("REVENUE DECLINE MODEL: INITIAL EXPERIMENT")
    results.append("=" * 48)
    results.append(f"Total rows: {len(df)}")
    results.append(f"Training rows: {len(train)}")
    results.append(f"Test rows: {len(test)}")
    results.append(f"Features: {available_features}")
    results.append(f"Training dates: {train['Date'].min().date()} to "
                   f"{train['Date'].max().date()}")
    results.append(f"Test dates: {test['Date'].min().date()} to "
                   f"{test['Date'].max().date()}")

    results.append("\nTraining target counts:")
    results.append(y_train.value_counts().sort_index().to_string())

    results.append("\nTest target counts:")
    results.append(y_test.value_counts().sort_index().to_string())

    results.append("\n--- Majority-class baseline ---")
    results.append(
        f"Accuracy: {accuracy_score(y_test, dummy_predictions):.3f}"
    )
    results.append(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, dummy_predictions):.3f}"
    )

    results.append("\n--- Logistic Regression ---")
    results.append(
        f"Accuracy: {accuracy_score(y_test, model_predictions):.3f}"
    )
    results.append(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, model_predictions):.3f}"
    )
    results.append("\nClassification report:")
    results.append(
        classification_report(
            y_test,
            model_predictions,
            labels=[0, 1],
            target_names=["No revenue decline", "Revenue decline"],
            zero_division=0,
        )
    )

    results.append("Confusion matrix (actual rows, predicted columns):")
    results.append(
        str(confusion_matrix(y_test, model_predictions, labels=[0, 1]))
    )

    results.append(
        "\nIMPORTANT: This is a small proof-of-concept experiment, "
        "not a validated investment-risk predictor."
    )
    results.append(
        "The chronological split is by report date (no date is in both "
        "sets), not by company. Company observations may appear in both sets."
    )
    results.append(
        "For the full time- and company-aware evaluation with baselines, "
        "see model_evaluation.md (src/ml/evaluate_revenue_decline.py)."
    )
    results.append(
        "Future outcome columns, Date, and Ticker were not used "
        "as model features."
    )

    report = "\n".join(results)

    print(report)

    # Do not overwrite an earlier report.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print("\nReport already exists; it was not overwritten:")
        print(OUTPUT_FILE)
        print("\nDelete or rename that report manually if you want "
              "to generate a new one.")
        return

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print("\nReport saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
