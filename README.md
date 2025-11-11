# ML Framework for Customer Conversion Prediction

HubSpot assessment

## Overview

This framework implements a complete ML pipeline from data ingestion to model serving, with emphasis on:
- Data validation at multiple stages
- Configurable experimentation
- MLflow experiment tracking
- Unit testing
- Automated documentation

## Architecture

### Core Components

**Data Pipeline** (`src/ml_framework/data/`)
- `loader.py`: Data ingestion with schema validation
- `preprocessor.py`: Feature engineering and data cleaning
- Handles duplicates using "most complete" strategy
- Schema validation catches data quality issues early

**Models** (`src/ml_framework/models/`)
- Abstract base class for model interface consistency
- Implementations: RandomForest, XGBoost, Logistic Regression
- Feature importance extraction
- Probability calibration

**Training** (`src/ml_framework/training/`)
- Configurable training pipeline
- MLflow integration for experiment tracking
- Automated artifact generation
- Performance metrics logging

**Serving** (`src/ml_framework/serving/`)
- FastAPI API
- Pydantic validation for request/response
- Single and batch prediction endpoints
- Health monitoring

**Utilities** (`src/ml_framework/utils/`)
- YAML configuration management
- Structured logging
- Helper functions

## Data Validation Strategy

Three-layer validation approach:

1. **Schema Validation** (Data Loading)
   - Validates column presence and types
   - Checks for required fields
   - Identifies duplicate IDs
   - Removes invalid records (e.g., MRR ≤ 0 for customers)

2. **API Validation** (Pydantic)
   - Request/response schema enforcement
   - Type checking
   - Business rule validation (e.g., ALEXA_RANK > 0)

3. **Unit Tests** (pytest)
   - Business logic validation
   - Pipeline behavior verification
   - Feature engineering correctness
   - Data integrity checks

## Quick Start

### View Documentation
```bash
start docs/build/html/index.html
```

### Installation
```bash
pip install -r requirements.txt
```

### Training a Model
```bash
# Using the notebook
jupyter notebook notebooks/demo.ipynb

# Or via Python
python -c "from ml_framework.training import Trainer; from ml_framework.utils import load_config; trainer = Trainer(load_config('configs/config.yaml')); trainer.train()"
```

### Starting Services

**MLflow UI:**
```bash
cd ml-framework-package
mlflow ui --port 5000
# Access at http://localhost:5000
```

**API Server:**
```bash
cd ml-framework-package
python run_api.py
# API docs at http://localhost:8000/docs
```

### Running Tests
```bash
pytest
```

## Configuration

All settings are controlled via `configs/config.yaml`:
```yaml
experiment:
  name: customer_conversion_baseline
  mlflow_tracking_uri: mlruns

data:
  company_data_path: data/customers.csv
  activity_data_path: data/usage_actions.csv
  test_size: 0.2

model:
  type: logistic_regression
  hyperparameters:
    C: 1.0
    max_iter: 1000

features:
  categorical: [EMPLOYEE_RANGE, INDUSTRY]
  numerical: [ALEXA_RANK, total_actions, total_users]

reproducibility:
  seed: 42
```

## Project Structure
```
ml-framework-package/
├── src/ml_framework/       # Core framework code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model implementations
│   ├── training/           # Training pipeline
│   ├── serving/            # API and prediction serving
│   └── utils/              # Configuration and logging
├── configs/                # Configuration files
├── data/                   # Raw data files
├── artifacts/              # Trained models and artifacts
├── mlruns/                 # MLflow experiment tracking
├── notebooks/              # Jupyter notebooks for demos
├── tests/                  # Unit and integration tests
├── run_api.py              # API server launcher
└── requirements.txt        # Python dependencies
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "id": 123,
    "ALEXA_RANK": 50000,
    "EMPLOYEE_RANGE": "26 to 50",
    "INDUSTRY": "COMPUTER_SOFTWARE",
    "total_actions": 150,
    "total_users": 5,
    "days_active": 30,
    "activity_frequency": 5.0
  }'
```

Response:
```json
{
  "company_id": 123,
  "conversion_probability": 0.7543,
  "prediction": 1,
  "confidence": "high"
}
```

## Testing

Tests validate:
- Data quality (no invalid values)
- Business rules (e.g., ACTIONS >= USERS)
- Pipeline behavior (removes bad data correctly)
- Feature engineering (creates required features)

Run with:
```bash
pytest                          # All tests
pytest -v                       # Verbose output
pytest -k "pipeline"            # Specific tests
pytest --html=report.html       # HTML report
```

## Code Quality

Automated linting and formatting:
```bash
# Install pre-commit hooks
pre-commit install

```

Pre-commit hooks run automatically on `git commit`.

## MLflow Tracking

All experiments are tracked in MLflow:
- Hyperparameters
- Metrics (accuracy, precision, recall, F1)
- Artifacts (models, plots, reports)
- Run comparisons

View experiments:
```bash
mlflow ui --port 5000
```

## Model Artifacts

Each training run generates:
```
artifacts/customer_conversion_baseline_YYYYMMDD_HHMMSS/
├── models/
│   ├── model.joblib
│   ├── feature_engineer.joblib
│   ├── employee_encoder.joblib
│   ├── industry_encoder.joblib
│   └── missing_handler.joblib
├── metrics/
│   └── classification_report.txt
├── plots/
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── config.yaml
```

## Development

### Adding a New Model

1. Extend `BaseModel` in `src/ml_framework/models/base_model.py`
2. Implement required methods: `fit`, `predict`, `predict_proba`, `get_feature_importance`
3. Add model type to `config.yaml`
4. Register in model factory

### Adding Features

1. Modify `DataPreprocessor._create_activity_features()`
2. Update feature list in `config.yaml`
3. Add validation tests

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Author

Jahnavi Gajula
