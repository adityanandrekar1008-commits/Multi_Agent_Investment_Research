
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "revenue_decline_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "company_wise_validation_results.csv"
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

    required = ["Ticker", TARGET] + FEATURES
    missing_columns = [
        column for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.dropna(subset=["Ticker", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    companies = sorted(df["Ticker"].unique())

    # Keep preprocessing inside the pipeline so that the imputer
    # and scaler are fitted only on each training fold.
    model_template = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    results = []
    all_actual = []
    all_predicted = []

    print("\n========== COMPANY-WISE VALIDATION ==========")
    print("Method: Leave-One-Company-Out")
    print("Companies:", len(companies))
    print("Rows:", len(df))

    for company in companies:
        train = df[df["Ticker"] != company]
        test = df[df["Ticker"] == company]

        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]

        # Both classes must be present in training.
        if y_train.nunique() < 2:
            print(
                f"\nSkipping {company}: training data "
                "contains only one target class."
            )
            continue

        model = clone(model_template)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        # Balanced accuracy is undefined when the test fold
        # contains only one class, so report it as unavailable.
        if y_test.nunique() == 2:
            balanced_accuracy = balanced_accuracy_score(
                y_test, predictions
            )
        else:
            balanced_accuracy = float("nan")

        matrix = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        results.append({
            "Test_Company": company,
            "Train_Rows": len(train),
            "Test_Rows": len(test),
            "Actual_Declines": int((y_test == 1).sum()),
            "Predicted_Declines": int((predictions == 1).sum()),
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
            "TN": int(matrix[0, 0]),
            "FP": int(matrix[0, 1]),
            "FN": int(matrix[1, 0]),
            "TP": int(matrix[1, 1]),
        })

        all_actual.extend(y_test.tolist())
        all_predicted.extend(predictions.tolist())

        print(
            f"\nCompany: {company}"
            f"\n  Training rows: {len(train)}"
            f"\n  Test rows: {len(test)}"
            f"\n  Actual declines: {(y_test == 1).sum()}"
            f"\n  Predicted declines: {(predictions == 1).sum()}"
            f"\n  Accuracy: {accuracy:.3f}"
            f"\n  Balanced accuracy: "
            f"{balanced_accuracy:.3f}"
            if y_test.nunique() == 2
            else
            f"\nCompany: {company}"
            f"\n  Training rows: {len(train)}"
            f"\n  Test rows: {len(test)}"
            f"\n  Actual declines: {(y_test == 1).sum()}"
            f"\n  Predicted declines: {(predictions == 1).sum()}"
            f"\n  Accuracy: {accuracy:.3f}"
            f"\n  Balanced accuracy: unavailable "
            "(test fold contains one class)"
        )

    if not results:
        raise ValueError("No company folds could be evaluated.")

    results_df = pd.DataFrame(results)

    print("\n========== RESULTS SUMMARY ==========")
    print(results_df.to_string(index=False))

    print("\nOverall pooled accuracy:",
          f"{accuracy_score(all_actual, all_predicted):.3f}")

    overall_matrix = confusion_matrix(
        all_actual, all_predicted, labels=[0, 1]
    )
    print("\nPooled confusion matrix (actual rows, predicted columns):")
    print(overall_matrix)

    print(
        "\nIMPORTANT: This is a small exploratory dataset. "
        "These results do not validate a real investment-risk predictor."
    )

    # Never overwrite an earlier results file.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print("\nOutput already exists; no file was overwritten:")
        print(OUTPUT_FILE)
        print("Rename or remove the old report manually to save a new one.")
    else:
        results_df.to_csv(OUTPUT_FILE, index=False)
        print("\nResults saved to:", OUTPUT_FILE)

    print("\nOriginal datasets and training scripts were not modified.")


if __name__ == "__main__":
    main()
