
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "revenue_decline_features.csv"
)

OUTPUT_FILE = (
    ROOT / "reports"
    / "date_grouped_validation_results.txt"
)

TARGET = "Next_Period_Revenue_Decline"

FEATURES = [
    "Net_Income_Margin",
    "Operating_Margin",
]


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def main():
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Keep every company from a given date in the same partition.
    dates = sorted(df["Date"].unique())

    if len(dates) < 3:
        raise ValueError("Not enough distinct dates for this validation.")

    split_index = int(len(dates) * 0.70)
    split_index = max(1, min(split_index, len(dates) - 1))
    cutoff_date = dates[split_index]

    train = df[df["Date"] < cutoff_date].copy()
    test = df[df["Date"] >= cutoff_date].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test partition is empty.")

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    lines = [
        "DATE-GROUPED CHRONOLOGICAL VALIDATION",
        "=" * 45,
        f"Total observations: {len(df)}",
        f"Distinct reporting dates: {len(dates)}",
        f"Training observations: {len(train)}",
        f"Test observations: {len(test)}",
        f"Training dates: {train['Date'].min().date()} to "
        f"{train['Date'].max().date()}",
        f"Test dates: {test['Date'].min().date()} to "
        f"{test['Date'].max().date()}",
        f"Features: {FEATURES}",
        "",
        "Training target counts:",
        y_train.value_counts().sort_index().to_string(),
        "",
        "Test target counts:",
        y_test.value_counts().sort_index().to_string(),
    ]

    if y_train.nunique() < 2:
        lines.append(
            "\nCannot train Logistic Regression: training data "
            "contains only one target class."
        )
    else:
        model = make_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        lines.extend([
            "\n--- Logistic Regression ---",
            f"Accuracy: {accuracy_score(y_test, predictions):.3f}",
            f"Balanced accuracy: "
            f"{balanced_accuracy_score(y_test, predictions):.3f}",
            "Confusion matrix (actual rows, predicted columns):",
            str(confusion_matrix(y_test, predictions, labels=[0, 1])),
        ])

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_predictions = baseline.predict(X_test)

        lines.extend([
            "\n--- Majority-class baseline ---",
            f"Accuracy: "
            f"{accuracy_score(y_test, baseline_predictions):.3f}",
            f"Balanced accuracy: "
            f"{balanced_accuracy_score(y_test, baseline_predictions):.3f}",
            str(confusion_matrix(
                y_test, baseline_predictions, labels=[0, 1]
            )),
        ])

    lines.extend([
        "",
        "Limitations:",
        "- Small dataset; results are exploratory.",
        "- Reporting dates may not equal publication dates.",
        "- This target measures next-report revenue decline, not bankruptcy.",
        "- Test rows are later dates, not necessarily unseen companies.",
    ])

    report = "\n".join(lines)
    print(report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print("\nExisting report not overwritten:", OUTPUT_FILE)
    else:
        OUTPUT_FILE.write_text(report, encoding="utf-8")
        print("\nReport saved to:", OUTPUT_FILE)

    print("Existing datasets and reports were not modified.")


if __name__ == "__main__":
    main()
