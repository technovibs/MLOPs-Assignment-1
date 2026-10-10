"""Preprocessing pipeline for model reproducibility."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessing_pipeline():
    """Create preprocessing pipeline."""
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def get_feature_names():
    """Return feature names grouped by type."""
    return {
        "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    }
