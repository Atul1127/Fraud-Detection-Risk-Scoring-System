# Fraud Detection & Risk Scoring System

> **End-to-end, time-aware fraud detection with ensemble ML, cost-sensitive decisioning, online historical features, explainability, experiment tracking, API serving, monitoring, Docker, and CI/CD.**

[![CI](https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Online%20Features-47A248?logo=mongodb&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)

FraudX is a production-style fraud detection platform built on the IEEE-CIS Fraud Detection dataset. It combines chronological evaluation, causal historical/velocity features, imbalanced learning, XGBoost/LightGBM/CatBoost ensembles, Optuna tuning, validation-only cost-sensitive threshold selection, SHAP explainability, MongoDB-backed online historical features, FastAPI inference, MLflow tracking, PSI monitoring, Docker Compose, and GitHub Actions CI/CD.

---

## Recruiter Snapshot

- **590,540 transactions** processed from the IEEE-CIS dataset.
- **Strict chronological 70/15/15 train/validation/test evaluation** with the final test period kept untouched.
- **Validation-selected cost threshold:** `0.347`, using configurable false-positive and false-negative costs of `1:10`.
- **Final untouched-test ensemble:** **ROC-AUC 0.8480**, **PR-AUC 0.4283**, **F1 0.3792**.
- **20 automated tests passing** in the verified local suite.
- XGBoost + LightGBM + CatBoost weighted ensemble with **35/35/30** weights.
- Training-only SMOTE; validation and final test data remain untouched.
- Real-time inference through FastAPI with MongoDB-backed historical features and prediction persistence.
- MLflow tracks configuration, validation metrics, final test metrics, thresholding and model artifacts.
- PSI-based monitoring covers prediction-score and online numeric-feature drift.
- Docker Compose and GitHub Actions provide reproducible local deployment and CI validation.

> **Evaluation note:** The final test metrics are the primary reported benchmark. The threshold is selected on validation data and then frozen before test evaluation, avoiding test-set threshold optimization.

---

# Architecture

```text
                    GitHub Push / PR
                           │
                    GitHub Actions
                    ┌──────┴──────┐
                    │ Tests + CI  │
                    │ Docker Build│
                    └──────┬──────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
   Training Pipeline                    FastAPI :8001
          │                                 │
          ▼                                 ▼
 Feature Engineering                    MongoDB :27017
          │                                 │
          ▼                                 │
 Chronological Split                       │
    ┌─────┼─────┐                           │
    ▼     ▼     ▼                           │
  Train   Val  Final Test                   │
    │      │      │                          │
    ▼      │      │                          │
  SMOTE    │      │                          │
    │      │      │                          │
    └──────┼──────┘                          │
           ▼                                 │
   XGB + LightGBM + CatBoost                 │
           │                                 │
           ▼                                 │
    Weighted Ensemble ◄──────────────────────┘
           │
           ▼
 Validation Cost Optimization
           │
           ▼
     Freeze Threshold
           │
           ▼
  Untouched Final Test
           │
           ▼
      Fraud Probability
           │
      ┌────┴────┐
      ▼         ▼
    SHAP     Persistence
                │
                ▼
         Drift Monitoring

          ┌──────────────┐
          │    MLflow    │
          │ Runs/Metrics │
          │  Artifacts   │
          └──────────────┘
```

### Runtime services

| Service | Purpose | Host Port |
|---|---|---:|
| **FraudX API** | Real-time fraud scoring and monitoring | `8001` |
| **MongoDB** | Historical transactions and prediction persistence | `27017` |
| **MLflow** | Experiment tracking and artifacts | `5000` |

Port `8001` is used intentionally so FraudX can coexist with applications using port `8000`.

---

# Core Capabilities

| Capability | Implementation |
|---|---|
| Fraud classification | XGBoost + LightGBM + CatBoost |
| Time-aware evaluation | Strict chronological train/validation/test split |
| Imbalanced learning | SMOTE on training data + model class balancing |
| Hyperparameter optimization | Optuna, optimized for PR-AUC |
| Decision optimization | Validation-only cost-sensitive threshold selection |
| Explainable AI | SHAP |
| Online historical features | MongoDB |
| Model serving | FastAPI + Uvicorn |
| Experiment tracking | MLflow |
| Drift monitoring | PSI for prediction scores and online numeric features |
| Containerization | Docker + Docker Compose |
| Automated validation | Pytest |
| CI/CD | GitHub Actions |
| Interactive analysis | Streamlit |

---

# Model Evaluation

## Final untouched temporal test

The production benchmark uses three chronological periods:

- **70% training:** model fitting and SMOTE.
- **15% validation:** model evaluation and threshold selection.
- **15% final test:** completely untouched during training and threshold selection.

The validation-selected threshold is frozen before the final test is evaluated.

| Metric | Final test |
|---|---:|
| ROC-AUC | **0.8480** |
| PR-AUC | **0.4283** |
| Precision | — |
| Recall | — |
| **F1** | **0.3792** |
| Frozen threshold | **0.347** |

Only ROC-AUC, PR-AUC and F1 are reported here because these are the verified final benchmark values. No test-time threshold optimization is performed.

### Threshold objective

The default decision policy is cost-sensitive:

```text
False positive cost = 1
False negative cost = 10
```

The threshold minimizing classification cost is selected **on validation data only**. This reflects a fraud setting where missed fraud can be materially more expensive than an unnecessary alert/review.

### Historical validation benchmark

An earlier validation-only run produced a weighted-ensemble ROC-AUC of `0.8921`, PR-AUC of `0.4857`, and F1 of `0.5081`. Those numbers are retained as historical context, **not as the final generalization benchmark**, because the validation threshold was optimized on that same validation period.

> Results can vary with library versions, cached artifacts, configuration and feature changes. The final untouched-test benchmark is the primary result.

---

# Machine Learning Pipeline

```text
IEEE-CIS Transactions + Identity
              │
              ▼
       Data Loading / Merge
              │
              ▼
   Time-Ordered Feature Engineering
              │
              ▼
     Chronological 70/15/15 Split
       ┌──────┼──────┐
       ▼      ▼      ▼
     Train    Val   Test
       │      │      │
       ▼      │      │
     SMOTE    │      │
       │      │      │
       └──────┼──────┘
              ▼
       XGB / LGBM / CatBoost
              │
              ▼
       Weighted Probability
              │
              ▼
  Cost Threshold on Validation
              │
              ▼
       Freeze Threshold
              │
              ▼
     Final Test Evaluation
```

## Feature Engineering

FraudX generates transaction-level signals including:

- Log-transformed transaction amount.
- Hour and day-of-week features.
- Time since previous transaction for a card.
- Historical card and combination frequencies.
- Card-level transaction velocity over configurable windows.
- Card/hour historical activity.
- Email-domain mismatch signals.
- M1–M9 match indicators.
- Selected Vesta features based on missingness.
- Missing-value-aware categorical encoding.

Time-dependent count and velocity features are calculated in transaction order and use historical context rather than future rows, reducing temporal leakage risk.

## Imbalanced Learning

FraudX applies **SMOTE only to the training split**. Validation and final test distributions are left untouched. The boosting models also use configured class-balancing mechanisms.

## Ensemble

```text
XGBoost  ── 35% ──┐
LightGBM ── 35% ──┼──► Fraud Probability
CatBoost ── 30% ──┘
```

Optuna optimizes PR-AUC. The decision threshold is selected separately on validation data using the configured cost function.

---

# Online Features with MongoDB

MongoDB provides historical context during online inference rather than acting only as a persistence layer.

```text
Incoming transaction
        │
        ▼
     FastAPI
        │
        ▼
 MongoDB history
        │
        ▼
Historical frequency / velocity features
        │
        ▼
Model preprocessing
        │
        ▼
Ensemble prediction
        │
        ▼
Persist transaction + prediction
```

Historical queries use earlier transactions, preserving the temporal nature of the feature pipeline.

---

# FastAPI

## Swagger

After starting Docker Compose:

**http://127.0.0.1:8001/docs**

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service/model/MongoDB health |
| `GET` | `/model/info` | Model version, threshold, weights, feature count |
| `GET` | `/monitoring/predictions` | Prediction volume, fraud rate and score drift |
| `GET` | `/monitoring/features` | Online numeric-feature distribution drift |
| `POST` | `/predict` | Score and persist a transaction |

### Example prediction request

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

The API uses the persisted production threshold from `config.yaml` for classification.

---

# Drift Monitoring

FraudX includes a lightweight production-style monitoring layer based on **Population Stability Index (PSI)**.

### Prediction monitoring

```http
GET /monitoring/predictions?window_hours=24
```

Tracks prediction volume, fraud prediction rate, average fraud probability, score-distribution changes and PSI drift level.

### Feature monitoring

```http
GET /monitoring/features?window_hours=24
```

Compares online numeric feature distributions across adjacent time windows.

### PSI interpretation

| PSI | Status |
|---:|---|
| `< 0.10` | Stable |
| `0.10–0.25` | Warning |
| `>= 0.25` | Drift |

The monitoring endpoints return `insufficient_data` until both comparison windows contain enough observations.

See [`docs/MONITORING.md`](docs/MONITORING.md) for implementation details and limitations.

---

# MLflow

FraudX tracks training runs with MLflow.

**Experiment:** `FraudX-Fraud-Detection`

Tracked information includes:

- Training configuration and model parameters.
- Validation ROC-AUC, PR-AUC, precision, recall and F1.
- Validation-selected threshold and threshold objective.
- Final untouched-test ROC-AUC, PR-AUC, precision, recall and F1.
- Training reports and model artifacts.

Validation and final test metrics are logged with separate `validation_*` and `test_*` metric names so the final benchmark cannot be confused with threshold-tuning metrics.

Open the local UI at:

**http://127.0.0.1:5000**

---

# Docker Deployment

FraudX runs as three local services:

```text
┌─────────────────────────────────────────────┐
│              Docker Compose                 │
│                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────┐│
│  │ FraudX API │  │  MongoDB   │  │ MLflow ││
│  │   :8001    │  │   :27017   │  │ :5000  ││
│  └────────────┘  └────────────┘  └────────┘│
└─────────────────────────────────────────────┘
```

### Start

```bash
git clone https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System.git
cd Fraud-Detection-Risk-Scoring-System
docker compose up --build -d
```

### Check

```bash
docker compose ps
```

Expected services:

```text
fraudx-api       Up
fraudx-mlflow    Up (healthy)
fraudx-mongodb   Up (healthy)
```

### Stop

```bash
docker compose down
```

### Local endpoints

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8001/docs |
| FastAPI health | http://127.0.0.1:8001/health |
| MLflow | http://127.0.0.1:5000 |
| MongoDB | `mongodb://127.0.0.1:27017` |

---

# CI/CD

GitHub Actions runs on pushes and pull requests to `main`.

```text
Push / Pull Request
        │
        ▼
Checkout → Python 3.12 → Install dependencies
        │
        ▼
Compile sources → Pytest → Docker Compose validation
        │
        ▼
Docker Build
        │
        └── main push → publish FraudX API image to GHCR
```

The workflow also uses GitHub Actions cache for Docker builds.

---

# Testing

The automated suite covers:

- Strict chronological train/validation/test boundaries.
- Historical frequency and transaction-time feature behavior.
- Numeric feature output after preprocessing.
- Evaluation metrics and threshold contracts.
- Cost-sensitive threshold selection.
- Ensemble probability shape and bounds.
- API/project configuration.
- MongoDB/online feature behavior.
- Drift/PSI calculations.
- Docker Compose configuration.

**Verified local result: `20 passed`.**

Run locally:

```bash
pytest -q
```

---

# Explainability

SHAP is integrated for model interpretation, including:

- Global feature importance.
- Per-transaction explanations.
- Waterfall-style explanations.
- Feature contribution analysis.

The objective is not only to predict fraud but also to provide evidence for why a transaction received a high fraud score.

---

# Streamlit Dashboard

Run the interactive analysis dashboard with:

```bash
streamlit run app/streamlit_app.py
```
