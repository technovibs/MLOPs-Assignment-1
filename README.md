# Heart Disease Prediction - MLOps Assignment

End-to-end Machine Learning Operations pipeline for predicting heart disease risk using modern MLOps best practices.

## Quick Start

### Prerequisites
- Python 3.9+
- pip
- Docker
- Kubernetes tools (optional for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/technovibs/MLOPs-Assignment-1.git
cd MLOPs-Assignment-1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
python data/get_data.py
```

5. Train the model:
```bash
python models/train.py
```

6. Run tests:
```bash
pytest tests/
```

7. Start the API:
```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## Project Structure

```text
MLOPs-Assignment-1/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── test_app.py
├── data/
│   ├── get_data.py
│   └── raw/
├── models/
│   ├── preprocessing.py
│   ├── train.py
│   └── best_model_pipeline.pkl
├── notebooks/
│   └── 01_EDA.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
├── .github/workflows/
│   └── ci_cd.yml
├── Dockerfile
├── requirements.txt
├── REPORT.md
├── .gitignore
├── .dockerignore
├── README.md
└── LICENSE
```

## API Usage

### Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example response

```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "confidence": 0.85
}
```

## Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

## Kubernetes

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## CI/CD

The repository includes a GitHub Actions workflow for:
- dependency installation
- linting
- unit tests
- model training
- Docker image build

## Monitoring

Prometheus metrics are exposed via FastAPI instrumentor at:
```text
http://localhost:8000/metrics
```

## Assignment Coverage

- Data acquisition and EDA
- Feature engineering and model training
- MLflow experiment tracking
- Model packaging and reproducibility
- CI/CD workflow
- Containerization
- Kubernetes deployment
- Monitoring and logging
- Documentation and reporting

## Repository

https://github.com/technovibs/MLOPs-Assignment-1
