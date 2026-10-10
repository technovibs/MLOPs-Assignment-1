"""Pydantic schemas for heart disease prediction API."""

from typing import Optional

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Input schema for patient attributes."""

    age: int = Field(..., ge=18, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=80, le=250)
    chol: int = Field(..., ge=0, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=60, le=220)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=10.0)
    slope: int = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0.0, le=4.0)
    thal: int = Field(..., ge=0, le=3)

    model_config = {
        "json_schema_extra": {
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
    }


class PredictionResponse(BaseModel):
    """API response for predictions."""

    prediction: int
    risk_label: str
    confidence: float


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    model_loaded: bool
    version: Optional[str] = None
