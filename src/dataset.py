from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path

# Load dataset
data = load_breast_cancer()

# Convert into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

# Save dataset
output_path = Path("data") / "tumor_dataset.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print("Saved at:", output_path)
print(df.head())