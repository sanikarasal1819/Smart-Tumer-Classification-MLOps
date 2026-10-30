import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

# Load processed data
X_train, X_test, y_train, y_test = joblib.load(
    "artifacts/processed_data.joblib"
)

# Load trained model
model = joblib.load("artifacts/best_model.joblib")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

display.plot()
plt.title("Tumor Classification - Confusion Matrix")
plt.savefig("artifacts/confusion_matrix.png")
plt.show()

print("Confusion Matrix saved successfully!")