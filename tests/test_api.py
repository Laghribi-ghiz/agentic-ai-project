import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

import api.main as api_main


@pytest.fixture
def client(monkeypatch):
    model = DummyClassifier(strategy="constant", constant="No")
    model.fit([[0], [1]], ["No", "Yes"])

    monkeypatch.setattr(api_main, "model", model)

    return TestClient(api_main.app)


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict(client):
    payload = {
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
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in ["Yes", "No"]
    assert 0 <= result["probability_no"] <= 1
    assert 0 <= result["probability_yes"] <= 1