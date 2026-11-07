"""Linear Regression: House Price Prediction (Assignment 1 and Mini Project 2)."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "house_prices" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    if not DATA_PATH.exists():
        from download_datasets import main as download_datasets
        download_datasets()
    if not DATA_PATH.exists():
        sys.exit("Dataset download failed. Follow the manual download instructions in README.md.")

    data = pd.read_csv(DATA_PATH)
    if "SalePrice" not in data.columns:
        sys.exit("The housing CSV must contain a 'SalePrice' target column.")

    # The target is the value the model must learn to predict.
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    y = data["SalePrice"]

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
        ("regression", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    OUTPUT_DIR.mkdir(exist_ok=True)
    report = (
        "HOUSE PRICE PREDICTION – LINEAR REGRESSION\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Mean Squared Error (MSE): {mse:,.2f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:,.2f}\n"
        f"R² score: {r2:.4f}\n"
    )
    print(report)
    (OUTPUT_DIR / "house_price_metrics.txt").write_text(report, encoding="utf-8")

    # Plot actual and predicted prices for the held-out test set.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.65)
    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual house price")
    plt.ylabel("Predicted house price")
    plt.title("Actual vs Predicted House Prices – Linear Regression")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "house_prices_actual_vs_predicted.png", dpi=300)
    plt.close()

    print(f"Metrics saved to: {OUTPUT_DIR / 'house_price_metrics.txt'}")
    print(f"Plot saved to: {OUTPUT_DIR / 'house_prices_actual_vs_predicted.png'}")


if __name__ == "__main__":
    main()
