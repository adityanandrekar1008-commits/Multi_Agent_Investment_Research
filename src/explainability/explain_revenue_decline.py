
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "revenue_decline_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_FILE = REPORT_DIR / "revenue_decline_predictions.csv"
EXPLANATIONS_FILE = REPORT_DIR / "revenue_decline_feature_explanations.csv"

FEATURES = [
    "Net_Income_Margin",
    "Operating_Margin",
    "Debt_To_Assets",
    "Debt_To_Equity",
    "Current_Ratio",
]

TARGET = "Next_Period_Revenue_Decline"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    # Protect existing results.
    if PREDICTIONS_FILE.exists() or EXPLANATIONS_FILE.exists():
        raise FileExistsError(
            "An output file already exists. Rename or back it up first."
        )

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET])
    df = df.sort_values("Date").reset_index(drop=True)

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    split_index = int(len(df) * 0.70)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)
    X_test = test[FEATURES]

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both target classes.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]

    # Coefficients refer to standardized features.
    explanations = pd.DataFrame({
        "Feature": FEATURES,
        "Standardized_Coefficient": classifier.coef_[0],
    })
    explanations["Absolute_Coefficient"] = (
        explanations["Standardized_Coefficient"].abs()
    )
    explanations = explanations.sort_values(
        "Absolute_Coefficient", ascending=False
    )

    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    prediction_report = test[
        ["Date", "Ticker", TARGET]
    ].copy()
    prediction_report["Predicted_Revenue_Decline"] = predictions
    prediction_report["Model_Score"] = scores
    prediction_report["Correct_Prediction"] = (
        prediction_report[TARGET]
        == prediction_report["Predicted_Revenue_Decline"]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Write only new files; never overwrite earlier results.
    predictions_written = False
    try:
        prediction_report.to_csv(
            PREDICTIONS_FILE, index=False, mode="x"
        )
        predictions_written = True
        explanations.to_csv(
            EXPLANATIONS_FILE, index=False, mode="x"
        )
    except Exception:
        if predictions_written and PREDICTIONS_FILE.exists():
            PREDICTIONS_FILE.unlink()
        raise

    print("Explainability run completed.")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Prediction report: {PREDICTIONS_FILE}")
    print(f"Feature explanations: {EXPLANATIONS_FILE}")
    print("\nFeature coefficients:")
    print(explanations.to_string(index=False))
    print("\nPrediction preview:")
    print(prediction_report.to_string(index=False))
    print(
        "\nLimitations: small proof-of-concept dataset; "
        "coefficients are associations, not causes; model scores "
        "are not calibrated probabilities of financial distress."
    )


if __name__ == "__main__":
    main()
