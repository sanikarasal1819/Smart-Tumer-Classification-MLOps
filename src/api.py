from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI(title="Tumor Classification API")

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "best_model.joblib"
model = joblib.load(MODEL_PATH)


class TumorData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Tumor Classification API is running!"}


@app.post("/predict")
def predict(data: TumorData):
    if len(data.features) != 30:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly 30 feature values."
        )

    prediction = int(model.predict([data.features])[0])
    label = "malignant" if prediction == 0 else "benign"

    return {
        "prediction": prediction,
        "result": label,
        "note": "Educational demonstration only; not a medical diagnosis."
    } 