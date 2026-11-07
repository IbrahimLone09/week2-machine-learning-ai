"""Logistic Regression: Titanic Passenger Survival Classification."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    if not DATA_PATH.exists():
        from download_datasets import main as download_datasets
        download_datasets()
    if not DATA_PATH.exists():
        sys.exit("Dataset download failed. Follow the manual download instructions in README.md.")

    data = pd.read_csv(DATA_PATH)
    if "Survived" not in data.columns:
        sys.exit("The Titanic CSV must contain a 'Survived' target column.")

    # PassengerId is an identifier, not a useful passenger characteristic.
    X = data.drop(columns=["Survived", "PassengerId"], errors="ignore")
    y = data["Survived"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = (
        "TITANIC SURVIVAL PREDICTION – LOGISTIC REGRESSION\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "Confusion Matrix (rows = actual, columns = predicted):\n"
        f"{matrix}\n\n"
        "Classification Report:\n"
        f"{classification_report(y_test, predictions, zero_division=0)}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(report)
    (OUTPUT_DIR / "titanic_metrics.txt").write_text(report, encoding="utf-8")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=["Did not survive", "Survived"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Titanic Survival – Logistic Regression")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "titanic_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"Metrics saved to: {OUTPUT_DIR / 'titanic_metrics.txt'}")
    print(f"Plot saved to: {OUTPUT_DIR / 'titanic_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
