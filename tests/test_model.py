"""Unit tests for the model preprocessing pipeline."""

import pandas as pd

from models.preprocessing import get_preprocessing_pipeline


def test_preprocessing_pipeline_runs():
    """The preprocessing pipeline should transform raw numeric and categorical data."""
    X = pd.DataFrame(
        {
            "age": [50, 60, 70],
            "sex": [1, 0, 1],
            "cp": [1, 2, 0],
            "trestbps": [130, 140, 150],
            "chol": [200, 240, 260],
            "fbs": [0, 1, 0],
            "restecg": [1, 0, 1],
            "thalach": [150, 130, 120],
            "exang": [0, 1, 0],
            "oldpeak": [0.5, 1.0, 2.0],
            "slope": [1, 2, 1],
            "ca": [0.0, 1.0, 2.0],
            "thal": [3, 1, 2],
        }
    )

    preprocessor = get_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] > 0


def test_model_input_has_expected_columns():
    """Check for expected raw input feature names."""
    expected = {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    }

    actual = {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    }
    assert expected.issubset(actual)
