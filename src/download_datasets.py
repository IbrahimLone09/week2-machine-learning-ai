"""Download public mirror copies of the Kaggle training datasets."""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    ROOT / "data" / "house_prices" / "train.csv":
        "https://raw.githubusercontent.com/krishnaik06/Advanced-House-Price-Prediction-/master/train.csv",
    ROOT / "data" / "titanic" / "train.csv":
        "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv",
}


def main():
    for destination, url in DATASETS.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already present: {destination.relative_to(ROOT)}")
            continue
        print(f"Downloading {destination.relative_to(ROOT)} ...")
        urlretrieve(url, destination)
        print(f"Saved: {destination.relative_to(ROOT)}")
    print("Dataset download complete. If a download fails, use the official Kaggle links in README.md.")


if __name__ == "__main__":
    main()
