# Heart Disease Prediction - MLOps Assignment Report

## Overview
This project implements an end-to-end MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset. The system includes data acquisition, EDA, feature engineering, experiment tracking, model training, API serving, containerization, deployment manifests, and monitoring hooks.

## Dataset
- Source: UCI Machine Learning Repository
- Task: Binary classification
- Target: presence or absence of heart disease
- Data file: CSV generated from `data/get_data.py`

## Approach
- Load and clean the dataset
- Convert multiclass target into binary target
- Engineer feature transformations for numeric and categorical columns
- Train multiple models including logistic regression and random forest
- Track parameters and metrics with MLflow
- Save the best pipeline for inference
- Serve predictions through a FastAPI app
- Package in Docker
- Deploy using Kubernetes manifests

## Modeling
The deployment-ready pipeline combines preprocessing with the selected model to enable reliable inference.

## Deployment
The project includes sample Kubernetes manifests for deployment and a FastAPI service for serving predictions.

## Monitoring
Prometheus-compatible metrics are exposed via FastAPI instrumentation for monitoring API health and throughput.

## CI/CD
A GitHub Actions workflow performs setup, dependency installation, tests, training, and image build steps.
