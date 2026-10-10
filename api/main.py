import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "LogisticRegression.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Autonomous ML Engineer Agent API",
    description="Customer churn prediction API",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        data = pd.DataFrame([customer.model_dump()])

        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]

        return {
            "prediction": prediction,
            "probability_no": float(probabilities[0]),
            "probability_yes": float(probabilities[1]),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )