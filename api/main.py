import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "LogisticRegression.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


model = None


def get_model():
    global model

    if model is None:
        model = load_model()

    return model

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

        current_model = get_model()

        prediction = current_model.predict(data)[0]
        probabilities = current_model.predict_proba(data)[0]

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