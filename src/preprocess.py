import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Breast_cancer.csv")

# Remove unnecessary ID column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data
Path("artifacts").mkdir(exist_ok=True)

import joblib

joblib.dump(scaler, "artifacts/scaler.joblib")
joblib.dump(
    (X_train, X_test, y_train, y_test),
    "artifacts/processed_data.joblib"
)

print("Preprocessing completed successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])