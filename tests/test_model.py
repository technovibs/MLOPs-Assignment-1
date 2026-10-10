"""Unit tests for model training and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from models.preprocessing import get_preprocessing_pipeline


def test_preprocessing_pipeline_runs():
    """The preprocessing pipeline should transform raw numeric and categorical data."""
    data = pd.DataFrame(
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
    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == data.shape[0]
    assert transformed.shape[1] > 0


def test_model_training_and_prediction():
    """Confirm a model can train and generate predictions."""
    X = pd.DataFrame(
        {
            "age": np.random.randint(30, 70, 60),
            "sex": np.random.randint(0, 2, 60),
            "cp": np.random.randint(0, 4, 60),
            "trestbps": np.random.randint(100, 200, 60),
            "chol": np.random.randint(150, 350, 60),
            "fbs": np.random.randint(0, 2, 60),
            "restecg": np.random.randint(0, 3, 60),
            "thalach": np.random.randint(80, 180, 60),
            "exang": np.random.randint(0, 2, 60),
            "oldpeak": np.random.uniform(0, 5, 60),
            "slope": np.random.randint(0, 3, 60),
            "ca": np.random.uniform(0, 4, 60),
            "thal": np.random.randint(0, 4, 60),
        }
    )
    y = np.random.randint(0, 2, 60)

    pipeline = Pipeline(
        [
            ("preprocessor", get_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == (60,)
    assert set(predictions).issubset({0, 1})
