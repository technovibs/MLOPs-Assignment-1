"""
Preprocessing pipeline for Heart Disease dataset
Ensures reproducibility and consistency between training and inference
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def get_preprocessing_pipeline():
    """
    Create preprocessing pipeline for heart disease features
    
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    # Define feature types
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    
    # Numeric preprocessing: imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def get_feature_names():
    """Get feature names in correct order"""
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols
    }
