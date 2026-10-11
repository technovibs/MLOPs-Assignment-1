

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

# fetch dataset
heart_disease = fetch_ucirepo(id=45)
# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets
# metadata
print(heart_disease.metadata)
# variable information
print(heart_disease.variables)

def load_data():
    # Fetch dataset (ID 45 = Heart Disease)
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    # Convert multiclass target (0-4) to binary (0: no disease, 1: disease)
    y = y['num'].apply(lambda x: 1 if x > 0 else 0)
    return X, y

def perform_eda(X, y):
    df = pd.concat([X, y.rename('target')], axis=1)
    os.makedirs("artifacts", exist_ok=True)

    # Class Balance
    plt.figure(figsize=(6,4))
    sns.countplot(x='target', data=df)
    plt.title("Class Balance")
    plt.savefig("artifacts/class_balance.png")

    # Correlation Heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.savefig("artifacts/correlation_heatmap.png")

def train_and_track():
    X, y = load_data()
    perform_eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify column types
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    # Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    mlflow.set_experiment("Heart_Disease_Classification")

    best_model = None
    best_auc = 0

    for model_name, classifier in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', classifier)])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            # Log to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "roc_auc": auc})
            mlflow.log_artifact("artifacts/correlation_heatmap.png")
            # Avoid skops UntrustedTypesFoundException by using classic pickle serialization
            mlflow.sklearn.log_model(pipeline, "model", serialization_format="pickle")

            if auc > best_auc:
                best_auc = auc
                best_model = pipeline

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    # Save best model to disk for Docker image
    joblib.dump(best_model, 'models/best_model_pipeline.pkl')
    print(f"Training complete. Best model AUC: {best_auc}")

if __name__ == "__main__":
    train_and_track()
