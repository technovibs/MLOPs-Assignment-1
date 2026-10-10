from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Heart Disease Prediction API")
logging.basicConfig(level=logging.INFO)

# Load model pipeline
try:
    model = joblib.load('models/best_model_pipeline.pkl')
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Expose metrics for Prometheus
Instrumentator().instrument(app).expose(app)

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int

@app.post("/predict")
def predict(patient: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input to DataFrame
    df = pd.DataFrame([patient.dict()])
    
    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        logging.info(f"Prediction made: class={prediction}, prob={probability:.4f}")
        
        return {
            "prediction": int(prediction),
            "risk_label": "High Risk" if prediction == 1 else "Low Risk",
            "confidence": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
