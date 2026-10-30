from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "breast_cancer.csv"


def load_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame.to_csv(CSV_PATH, index=False)

    return frame, list(dataset.feature_names)


def split_dataset(test_size=0.2, random_state=42):
    frame, feature_names = load_dataset()

    X = frame[feature_names]
    y = frame["target"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


if __name__ == "__main__":
    frame, features = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset()

    print("Dataset created successfully!")
    print("Dataset shape:", frame.shape)
    print("Number of features:", len(features))
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Saved file:", CSV_PATH)