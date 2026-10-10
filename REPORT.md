# MLOps Heart Disease Prediction - Assignment Report

## 1. Executive Summary

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting heart disease risk based on patient health metrics. The solution demonstrates modern best practices in data processing, model development, experiment tracking, containerization, and production deployment.

### Key Deliverables:
- ✅ Data acquisition and exploratory analysis
- ✅ Multiple classification models with experiment tracking
- ✅ Reproducible ML pipeline with preprocessing
- ✅ Unit testing and CI/CD automation
- ✅ Docker containerization
- ✅ Kubernetes deployment manifests
- ✅ Monitoring and logging infrastructure

---

## 2. Project Structure

```
MLOPs-Assignment-1/
├── app/                           # FastAPI application
│   ├── main.py                   # API endpoints and logic
│   ├── schemas.py                # Pydantic request/response models
│   └── test_app.py               # API unit tests
├── data/                         # Data management
│   ├── get_data.py               # Dataset download and loading
│   └── raw/                      # Downloaded raw data (gitignored)
├── models/                       # Model training and storage
│   ├── train.py                  # Model training pipeline
│   ├── preprocessing.py          # Preprocessing configuration
│   └── best_model_pipeline.pkl   # Saved model (gitignored)
├── notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb   # Model development (optional)
├── tests/                        # Unit tests
│   ├── test_data.py             # Data processing tests
│   └── test_model.py            # Model training tests
├── k8s/                         # Kubernetes manifests
│   ├── deployment.yaml          # Deployment configuration
│   ├── service.yaml             # Service configuration
│   ├── configmap.yaml           # Environment configuration
│   └── ingress.yaml             # Ingress rules
├── monitoring/                  # Monitoring configurations
│   └── prometheus-config.yml    # Prometheus scrape config
├── scripts/                     # Utility scripts
│   ├── setup.sh                 # Environment setup
│   └── deploy-local-k8s.sh      # K8s deployment
├── .github/workflows/           # CI/CD pipelines
│   └── ci_cd.yml               # GitHub Actions workflow
├── Dockerfile                   # Container image definition
├── .dockerignore                # Docker build exclusions
├── .gitignore                   # Git exclusions
├── requirements.txt             # Python dependencies
├── README.md                    # Quick start guide
└── REPORT.md                    # This file
```

---

## 3. Data Acquisition & EDA (5 marks)

### Dataset
- **Source**: UCI Machine Learning Repository (Heart Disease Dataset, ID: 45)
- **Size**: 303 samples, 14 features + 1 target
- **Features**: Age, sex, chest pain type, blood pressure, cholesterol, and others
- **Target**: Binary classification (0: no disease, 1-4: disease present → binary: 0/1)

### Data Preparation
- **Script**: `data/get_data.py`
  - Automatic download from UCI ML Repository
  - Binary target conversion (multiclass → binary)
  - CSV exports for reproducibility
  - Class distribution analysis

### EDA Highlights
- Correlation heatmap: Identifies feature relationships
- Class balance distribution: Shows class representation
- Feature statistics: Age, blood pressure, cholesterol analysis
- Missing value handling: Median/mode imputation

**Artifacts Generated**:
- `artifacts/class_balance.png` - Target distribution
- `artifacts/correlation_heatmap.png` - Feature correlations

---

## 4. Feature Engineering & Model Development (8 marks)

### Feature Preprocessing
**File**: `models/preprocessing.py`

**Numeric Features** (6 features):
- Age, Resting BP, Cholesterol, Max Heart Rate, ST Depression, Vessels
- **Transformation**: Median imputation → StandardScaler

**Categorical Features** (7 features):
- Sex, Chest Pain Type, Fasting Blood Sugar, ECG Results, Exercise Angina, ST Slope, Thalassemia
- **Transformation**: Mode imputation → One-Hot Encoding

### Models Trained

1. **Logistic Regression**
   - Hyperparameters: max_iter=1000
   - Advantages: Interpretable, fast training
   - Use case: Baseline model

2. **Random Forest**
   - Hyperparameters: n_estimators=100, random_state=42
   - Advantages: Non-linear, feature importance
   - Use case: Production model (higher AUC)

### Evaluation Metrics
- **Accuracy**: Correct predictions ratio
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **ROC-AUC**: Area under ROC curve (best model selector)

### Model Selection
- Best model selected by ROC-AUC score
- Saved as `models/best_model_pipeline.pkl` using joblib
- Includes full preprocessing pipeline for reproducibility

---

## 5. Experiment Tracking (5 marks)

### MLflow Integration
**File**: `models/train.py`

**Tracked Information**:
- **Parameters**: Model type, hyperparameters
- **Metrics**: Accuracy, Precision, Recall, ROC-AUC
- **Artifacts**: Correlation heatmap, training plots
- **Model Registry**: Best model logged with pickle format

**Experiment Structure**:
```
Experiment: Heart_Disease_Classification
├── Run: LogisticRegression
│   ├── Metrics: {accuracy, precision, recall, roc_auc}
│   ├── Params: {model_type}
│   └── Artifacts: {correlation_heatmap.png}
└── Run: RandomForest
    ├── Metrics: {accuracy, precision, recall, roc_auc}
    ├── Params: {model_type}
    └── Artifacts: {correlation_heatmap.png}
```

**Access MLflow UI**:
```bash
mlflow ui
# Visit http://localhost:5000
```

---

## 6. Model Packaging & Reproducibility (7 marks)

### Packaging Strategy
- **Format**: sklearn pickle (.pkl)
- **Location**: `models/best_model_pipeline.pkl`
- **Contents**: Preprocessing + Classifier (full pipeline)

### Reproducibility Checklist
✅ Requirements file: All versions pinned
✅ Preprocessing pipeline: Saved with model
✅ Random seeds: Set in all models (random_state=42)
✅ Data download script: Automated UCI dataset fetch
✅ Environment file: requirements.txt

### Inference Consistency
```python
# Training time
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])
pipeline.fit(X_train, y_train)

# Inference time (API)
model = joblib.load('models/best_model_pipeline.pkl')
prediction = model.predict(input_data)  # Same preprocessing applied
```

---

## 7. CI/CD Pipeline & Automated Testing (8 marks)

### GitHub Actions Workflow
**File**: `.github/workflows/ci_cd.yml`

**Pipeline Stages**:

1. **Environment Setup**
   - Python 3.9 installation
   - Dependency installation
   - Flake8 linter installation

2. **Code Quality**
   - Linting: Flake8 checks
   - Excludes: E9, F63, F7, F82 errors

3. **Training**
   - Model training execution
   - Experiment tracking with MLflow
   - Metrics calculation

4. **Testing**
   - Unit tests: `pytest app/test_app.py`
   - Data validation tests
   - Model prediction tests

5. **Containerization**
   - Docker image build: `heart-disease-api:latest`
   - Image validation

6. **Artifact Storage**
   - Model artifacts archived
   - Available for download

**Triggers**:
- Push to main branch
- Pull requests to main branch

**Test Coverage**:
- **Data Tests** (`tests/test_data.py`):
  - Dataset download verification
  - Feature completeness
  - Target binary validation
  - Class balance checks

- **Model Tests** (`tests/test_model.py`):
  - Preprocessing pipeline creation
  - Output shape validation
  - Model training and prediction
  - Feature engineering

- **API Tests** (`app/test_app.py`):
  - Endpoint availability
  - Response validation
  - Prediction output format

---

## 8. Model Containerization (5 marks)

### Docker Configuration
**File**: `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY models/best_model_pipeline.pkl ./models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Image Details
- **Base**: python:3.9-slim (lightweight)
- **Size**: ~500MB (optimized)
- **Port**: 8000 (FastAPI default)
- **Entrypoint**: Uvicorn ASGI server

### Building and Running

**Build**:
```bash
docker build -t heart-disease-api:latest .
```

**Run Locally**:
```bash
docker run -p 8000:8000 heart-disease-api:latest
```

**Test Endpoint**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6
  }'
```

**Response**:
```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "confidence": 0.85
}
```

---

## 9. Production Deployment (7 marks)

### Kubernetes Manifests

#### 1. ConfigMap (`k8s/configmap.yaml`)
- Environment variables
- Configuration management
- Non-sensitive data

#### 2. Deployment (`k8s/deployment.yaml`)
- **Replicas**: 2 (high availability)
- **Image**: heart-disease-api:latest
- **Ports**: 8000
- **Resources**:
  - Requests: 256Mi memory, 250m CPU
  - Limits: 512Mi memory, 500m CPU
- **Health Monitoring**: Prometheus scrape annotations

#### 3. Service (`k8s/service.yaml`)
- **Type**: LoadBalancer
- **Selector**: app=heart-disease
- **Port Mapping**: 80 → 8000
- **Access**: External endpoint

#### 4. Ingress (`k8s/ingress.yaml`)
- **Class**: Nginx
- **TLS**: SSL certificate support
- **Host**: heart-disease.example.com
- **Path**: / → Service

### Deployment Instructions

**Prerequisites**:
```bash
# Install kubectl
# Install Minikube or use Docker Desktop K8s

# Start Minikube
minikube start
eval $(minikube docker-env)  # Use Minikube Docker
```

**Deploy**:
```bash
# Build and load image
docker build -t heart-disease-api:latest .

# Apply configurations
./scripts/deploy-local-k8s.sh

# Or manual deployment
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

**Access API**:
```bash
# Port forward
kubectl port-forward svc/heart-disease-service 8000:80

# Visit
http://localhost:8000/docs  # Swagger UI
```

**Verify Deployment**:
```bash
# Check pods
kubectl get pods -l app=heart-disease

# Check service
kubectl get svc heart-disease-service

# View logs
kubectl logs -f deployment/heart-disease-api
```

---

## 10. Monitoring & Logging (3 marks)

### Logging
**File**: `app/main.py`

```python
logging.basicConfig(level=logging.INFO)
logging.info(f"Prediction made: class={prediction}, prob={probability:.4f}")
```

**Log Output**:
- Request timestamps
- Prediction results
- Error messages
- Model loading status

**Kubernetes Log Access**:
```bash
kubectl logs -f deployment/heart-disease-api
```

### Metrics Collection
**Integration**: Prometheus FastAPI Instrumentator

```python
Instrumentator().instrument(app).expose(app)
```

**Exposed Metrics**:
- Request count
- Request duration
- Request size
- Response status codes

**Prometheus Scrape Config**: `monitoring/prometheus-config.yml`

**Access Metrics**:
```
http://localhost:8000/metrics
```

**Setup Prometheus**:
```bash
# Run Prometheus
docker run -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus-config.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### Grafana Dashboards
- Visualize metrics
- Create alerts
- Monitor API performance

---

## 11. Documentation (2 marks)

### README.md
Quick start guide with:
- Installation steps
- Data download
- Model training
- API usage
- Docker deployment

### REPORT.md (This File)
Comprehensive documentation including:
- Project architecture
- Component descriptions
- Setup instructions
- Deployment procedures
- Monitoring setup

### Code Documentation
- Docstrings in all modules
- Type hints in functions
- Comments for complex logic

---

## 12. Installation & Setup Instructions

### Quick Start

1. **Clone Repository**:
```bash
git clone https://github.com/technovibs/MLOPs-Assignment-1.git
cd MLOPs-Assignment-1
```

2. **Setup Environment**:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

3. **Download Dataset**:
```bash
python data/get_data.py
```

4. **Train Model**:
```bash
python models/train.py
```

5. **Run Tests**:
```bash
pytest tests/
pytest app/test_app.py
```

6. **Start API**:
```bash
uvicorn app.main:app --reload
# Visit http://localhost:8000/docs
```

7. **Build Docker Image**:
```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

8. **Deploy to Kubernetes**:
```bash
./scripts/deploy-local-k8s.sh
```

---

## 13. API Endpoints

### Health Check
```
GET /health
```

### Prediction
```
POST /predict
Content-Type: application/json

{
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
  "thal": 6
}
```

**Response**:
```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "confidence": 0.85
}
```

### Metrics
```
GET /metrics
```

### API Documentation
```
GET /docs           # Swagger UI
GET /redoc          # ReDoc
```

---

## 14. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Acquisition Layer                    │
│  UCI ML Repository → get_data.py → raw/ (features, target)  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  EDA & Analysis Layer                        │
│  Visualization → Correlation, Class Balance, Feature Stats  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Preprocessing & Feature Engineering             │
│  Imputation → Scaling (numeric) / Encoding (categorical)    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Model Training Layer                         │
│  Logistic Regression & Random Forest with Cross-Validation  │
│  MLflow Tracking: Metrics, Params, Artifacts               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Model Packaging & Storage                       │
│  Best Pipeline → best_model_pipeline.pkl (joblib)          │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼────┐    ┌──────▼────────┐  ┌──▼──────────┐
    │  Tests │    │  CI/CD Pipeline│  │  Container  │
    │ (Pytest)    │  (GitHub       │  │  (Docker)   │
    └────────┘    │   Actions)     │  └──┬──────────┘
                  └────────────────┘     │
                                         │
                  ┌──────────────────────▼──────────────────────┐
                  │     Docker Containerized API (FastAPI)      │
                  │  ├─ Model Loading                          │
                  │  ├─ Prediction Endpoint (/predict)         │
                  │  ├─ Prometheus Metrics (/metrics)          │
                  │  └─ Logging (INFO level)                   │
                  └──────────────────────┬──────────────────────┘
                                         │
        ┌────────────────┬──────────────┴────────┬────────────────┐
        │                │                       │                │
    ┌───▼────────┐  ┌────▼──────┐  ┌────────────▼──────┐  ┌───────▼──────┐
    │ Kubernetes │  │ ConfigMap │  │   Monitoring      │  │  Logging     │
    │ Deployment │  │ Management│  │ (Prometheus)      │  │  (App Logs)  │
    └────────────┘  └───────────┘  └───────────────────┘  └──────────────┘
```

---

## 15. Key Features & Highlights

✅ **Reproducibility**: Fixed random seeds, versioned dependencies
✅ **Scalability**: Kubernetes-ready, containerized, load-balanced
✅ **Monitoring**: Prometheus metrics, structured logging
✅ **Testing**: Unit tests for data, model, and API
✅ **CI/CD**: Automated testing and deployment pipeline
✅ **Documentation**: Code comments, docstrings, comprehensive report
✅ **Best Practices**: Type hints, error handling, configuration management

---

## 16. Performance Metrics

### Model Performance (Example)
| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 0.85 | 0.82 | 0.88 | 0.91 |
| Random Forest | 0.89 | 0.87 | 0.92 | 0.94 |

### API Performance
- Response time: ~50-100ms (depends on system)
- Throughput: ~10-20 req/sec (per pod)
- Memory usage: ~256-512MB per pod

---

## 17. Troubleshooting

### Common Issues

**Issue**: Model not found in Docker container
```
Solution: Ensure models/best_model_pipeline.pkl exists before building
Run: python models/train.py
Then: docker build -t heart-disease-api:latest .
```

**Issue**: Kubernetes pod not starting
```
Solution: Check logs
kubectl logs deployment/heart-disease-api
kubectl describe pod <pod-name>
```

**Issue**: Metrics not showing in Prometheus
```
Solution: Verify Prometheus scrape config
Check: http://localhost:8000/metrics is accessible
```

---

## 18. Future Enhancements

1. **Model Improvements**:
   - Hyperparameter tuning (GridSearchCV, Bayesian Optimization)
   - Ensemble methods
   - Deep learning models

2. **MLOps Enhancements**:
   - Model versioning and registry
   - A/B testing framework
   - Automated retraining pipeline

3. **Infrastructure**:
   - Helm charts for Kubernetes
   - Cloud deployment (GCP, AWS, Azure)
   - Database integration for prediction logging

4. **Monitoring**:
   - Advanced Grafana dashboards
   - Alert rules for anomalies
   - Data drift detection

---

## 19. References & Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Prometheus Documentation](https://prometheus.io/docs/)

---

## 20. Conclusion

This MLOps assignment demonstrates a complete end-to-end machine learning pipeline with production-grade infrastructure, automated testing, and monitoring. The implementation emphasizes reproducibility, scalability, and best practices in modern ML engineering.

**Repository**: https://github.com/technovibs/MLOPs-Assignment-1

---

**Last Updated**: 2026-10-10
**Author**: technovibs
**Status**: Complete
