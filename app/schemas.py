"""
Pydantic schemas for request and response validation
"""

from typing import Optional

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Patient health data for heart disease prediction."""
    age: int = Field(..., ge=18, le=120, description="Age of the patient")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: Female, 1: Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=80, le=250, description="Resting blood pressure")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of ST segment")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia")

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0.0,
                "thal": 6,
            }
        }


class PredictionResponse(BaseModel):
    """API response for the prediction endpoint."""
    prediction: int = Field(..., ge=0, le=1)
    risk_label: str
    confidence: float = Field(..., ge=0, le=1)


class HealthCheckResponse(BaseModel):
    """System health response."""
    status: str
    model_loaded: bool
    version: Optional[str] = None
