import joblib
import pandas as pd
from pathlib import Path

model_path = Path("models/LogisticRegression.joblib")

model = joblib.load(model_path)

sample = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}])

prediction = model.predict(sample)
probabilities = model.predict_proba(sample)

print("Model loaded successfully.")
print("Predicted churn:", prediction[0])
print("Churn probabilities:", probabilities[0])