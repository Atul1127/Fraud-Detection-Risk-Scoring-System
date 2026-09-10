# Fraud Detection & Risk Scoring System

> End-to-end, time-aware fraud detection with ensemble ML, cost-sensitive decisioning, causal historical features, explainability, FastAPI serving, MongoDB, MLflow, Docker, monitoring, and CI.

[![CI](https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-Online%20Features-47A248?logo=mongodb)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)

FraudX is a production-style fraud detection project built on the IEEE-CIS Fraud Detection dataset. It uses chronological evaluation, causal historical/velocity features, XGBoost + LightGBM + CatBoost, cost-sensitive thresholding, SHAP explainability, MongoDB-backed online features, FastAPI inference, MLflow tracking, PSI monitoring, Docker Compose, and GitHub Actions.

## Recruiter Snapshot

- **590,540 transactions** processed from the IEEE-CIS dataset.
- **Chronological 70/15/15 train/validation/test split** with the final test period kept untouched.
- **Frozen serving threshold:** `0.420`, selected on validation data using false-positive : false-negative costs of `1 : 10`.
- **Final untouched-test:** ROC-AUC **0.8394**, PR-AUC **0.4234**, Precision **0.2601**, Recall **0.5813**, F1 **0.3594**.
- XGBoost + LightGBM + CatBoost weighted ensemble with **35/35/30** weights.
- Training-only SMOTE; validation and final test remain untouched.
- FastAPI inference with MongoDB historical features and prediction persistence.
- MLflow tracking, SHAP explanations, PSI drift monitoring, Docker Compose, and CI.

> **Evaluation note:** The final test metrics are the primary benchmark. The threshold is selected on validation data and frozen before final test evaluation.

## Architecture

```text
GitHub Push / PR
       │
       ▼
GitHub Actions
       │
       ├── Tests
       └── Docker validation/build

Training Pipeline ──► Feature Engineering ──► Chronological Split
                                             ├── Train ──► SMOTE
                                             ├── Validation ──► threshold tuning
                                             └── Final Test ──► untouched benchmark
                                                        │
                                                        ▼
                                           XGB + LGBM + CatBoost
                                                        │
                                                        ▼
                                               Weighted Ensemble

FastAPI ──► MongoDB history ──► online features ──► model ──► prediction
   │                                                     │
   ├── /health                                            └── persistence
   ├── /model/info
   ├── /predict
   └── /monitoring/*

MLflow ──► runs / metrics / artifacts
SHAP    ──► model explanations
PSI     ──► prediction + feature drift
```

## Core Capabilities

| Capability | Implementation |
|---|---|
| Fraud classification | XGBoost + LightGBM + CatBoost |
| Evaluation | Strict chronological train/validation/test split |
| Imbalance handling | SMOTE on training data only + model class balancing |
| Tuning | Optuna, optimized for PR-AUC |
| Decisioning | Validation-only cost-sensitive threshold selection |
| Explainability | SHAP |
| Online features | MongoDB |
| Serving | FastAPI + Uvicorn |
| Tracking | MLflow |
| Monitoring | PSI drift |
| Deployment | Docker + Docker Compose |
| Testing | Pytest |
| CI/CD | GitHub Actions |

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System.git
cd Fraud-Detection-Risk-Scoring-System
```

### 2. Add the dataset

Download the IEEE-CIS Fraud Detection dataset and place these files in `data/raw/`:

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

The dataset is intentionally excluded from Git because of its size and Kaggle distribution restrictions.

### 3. Train

```bash
python train.py
```

Training generates processed features and the model checkpoint under `models/checkpoints/`.

### 4. Run the API stack

```bash
docker compose up --build -d
docker compose ps
```

Services:

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8001/docs |
| FastAPI health | http://127.0.0.1:8001/health |
| MLflow | http://127.0.0.1:5000 |
| MongoDB | `mongodb://127.0.0.1:27017` |

Stop with:

```bash
docker compose down
```

## API

### `GET /health`

Returns service, model, and MongoDB readiness.

### `GET /model/info`

Returns the loaded model's feature count, ensemble weights, and serving threshold.

### `POST /predict`

Example:

```json
{
  "transaction_id": "TX_TEST_001",
  "data": {
    "TransactionDT": 864000,
    "TransactionAmt": 149.99,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 555,
    "card3": 150,
    "card5": 226,
    "addr1": 315,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }
}
```

### Monitoring

```text
GET /monitoring/predictions?window_hours=24
GET /monitoring/features?window_hours=24
```

PSI is reported as stable (`< 0.10`), warning (`0.10–0.25`), or drift (`>= 0.25`).

## Model Evaluation

The production benchmark uses chronological periods:

- **70% training** — model fitting and SMOTE.
- **15% validation** — model evaluation and threshold selection.
- **15% final test** — completely untouched during training and threshold selection.

| Metric | Final test |
|---|---:|
| ROC-AUC | **0.8394** |
| PR-AUC | **0.4234** |
| Precision | **0.2601** |
| Recall | **0.5813** |
| F1 | **0.3594** |
| Frozen threshold | **0.420** |

### Threshold objective

```text
False-positive cost = 1
False-negative cost = 10
```

The threshold minimizing classification cost is selected on validation data only and then frozen before final test evaluation.

## Feature Engineering

FraudX includes:

- Log transaction amount.
- Hour and day-of-week.
- Time since previous transaction for a card.
- Historical card and combination frequencies.
- Card-level velocity features over configurable windows.
- Card/hour historical activity.
- Email-domain mismatch.
- M1–M9 match indicators.
- Selected Vesta features.
- Training-fitted categorical mappings.

Time-dependent counts and velocity features use prior transactions only.

## MLflow

Experiment: `FraudX-Fraud-Detection`

Tracked information includes training configuration, validation metrics, frozen threshold, final test metrics, reports, and model artifacts.

Open the local UI at http://127.0.0.1:5000 after starting Compose.

## Streamlit Dashboard

For the interactive analysis dashboard:

```bash
streamlit run app/streamlit_app.py
```

The dashboard requires trained checkpoints and processed validation data.

## Testing

Run:

```bash
pytest -q
```

The test suite covers temporal splitting, causal feature behavior, evaluation/thresholding, ensemble outputs, tuning helpers, monitoring, and project configuration.

## CI/CD

GitHub Actions runs on pushes and pull requests to `main` and performs:

```text
Install dependencies → compile sources → pytest → Docker Compose validation → Docker build
```

A push to `main` publishes the API image to GHCR.

## Project Structure

```text
.
├── api/                  # FastAPI + MongoDB online serving
├── app/                  # Streamlit dashboard + visualizations
├── data/                 # dataset instructions; raw/processed data ignored
├── docs/                 # monitoring documentation
├── src/
│   ├── data/             # loading and feature engineering
│   ├── models/           # ensemble implementation
│   ├── monitoring/       # PSI/drift monitoring
│   ├── evaluate.py       # metrics and threshold selection
│   ├── explain.py        # SHAP explanations
│   ├── mlflow_tracker.py # experiment tracking
│   ├── train.py          # training orchestration
│   └── tune.py           # Optuna tuning
├── tests/                # automated tests
├── config.yaml           # project configuration
├── Dockerfile
├── Dockerfile.mlflow
└── docker-compose.yml
```

## Limitations

This is a portfolio/research-oriented system rather than a hardened production fraud service. The repository does not provide authentication, rate limiting, secrets management, or a full production feature store. Model performance can vary with dependency versions, data preprocessing, and regenerated artifacts.
