# Week 2 – Machine Learning & AI

Complete implementation of the Week 2 assignments and Mini Project 2 from the provided assignment sheet.

## Included tasks

1. **Linear Regression – House Price Prediction:** train a regression model to predict `SalePrice`.
2. **Logistic Regression – Titanic Survival:** predict `Survived` (0 = did not survive, 1 = survived).
3. **Mini Project 2 – House Price Prediction Model:** evaluate with R² and plot predicted vs actual prices.

The housing assignment and mini project share one implementation because they use the same dataset and model.

## Project structure

```text
week2-ml-ai/
├── data/
│   ├── house_prices/       # train.csv is downloaded here
│   └── titanic/            # train.csv is downloaded here
├── outputs/                # metrics and plots generated here
├── src/
│   ├── download_datasets.py
│   ├── house_price_prediction.py
│   └── titanic_survival.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Datasets

The scripts automatically attempt to download public mirror copies of the Kaggle training CSVs when they are missing. Run this once if you want to download both datasets first:

```bash
python src/download_datasets.py
```

The sources are public GitHub mirrors of the competition training data:
- House Prices: `https://raw.githubusercontent.com/krishnaik06/Advanced-House-Price-Prediction-/master/train.csv`
- Titanic: `https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv`

Official competition pages and dataset descriptions:
- [Kaggle House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- [Kaggle Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

If automatic download fails, download `train.csv` from the corresponding official Kaggle page and place it at `data/house_prices/train.csv` or `data/titanic/train.csv` respectively. Kaggle sign-in or competition-rule acceptance may be required. The CSV files are not embedded in this ZIP because the runtime could not retrieve external dataset bytes here; the downloader is included to fetch them when you run the project with internet access. Check dataset/competition terms before redistribution.

## Run the housing assignment and mini project

```bash
python src/house_price_prediction.py
```

The program preprocesses missing numerical and categorical values, one-hot encodes categorical columns, trains Linear Regression, and reports MSE, RMSE and R². It saves:
- `outputs/house_price_metrics.txt`
- `outputs/house_prices_actual_vs_predicted.png`

## Run the Titanic assignment

```bash
python src/titanic_survival.py
```

The program preprocesses the data, trains Logistic Regression and reports accuracy, confusion matrix, precision, recall and F1-score. It saves:
- `outputs/titanic_metrics.txt`
- `outputs/titanic_confusion_matrix.png`

## Metrics explained

- **MSE:** average squared prediction error; lower is better.
- **RMSE:** square root of MSE, in the same units as house price.
- **R²:** proportion of target variation explained by the regression model. Closer to 1 is generally better; it can be negative.
- **Accuracy:** fraction of Titanic test examples classified correctly.
- **Confusion matrix:** counts correct and incorrect predictions for each class.

Scores are computed at runtime and are not fabricated or hard-coded.

