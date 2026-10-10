from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    sample_data = {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, 
        "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
