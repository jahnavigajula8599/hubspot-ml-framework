# Entity Relationship Diagram - HubSpot ML Framework

## Overview
This document provides comprehensive Entity Relationship Diagrams (ERD) for the HubSpot ML Framework, covering both data entities and ML pipeline components.

---

## 1. Data Model ERD

### Main Entities and Relationships

```mermaid
erDiagram
    COMPANY ||--o{ USAGE_ACTION : "generates"
    COMPANY ||--|| AGGREGATED_FEATURES : "has"
    COMPANY {
        int id PK
        float ALEXA_RANK
        string EMPLOYEE_RANGE
        string INDUSTRY
        int is_customer "Target variable (0 or 1)"
        datetime CLOSEDATE "Customers only"
        float MRR "Monthly Recurring Revenue (customers only)"
    }

    USAGE_ACTION {
        int id FK "References Company"
        datetime WHEN_TIMESTAMP
        int ACTIONS_CRM_CONTACTS
        int ACTIONS_CRM_COMPANIES
        int ACTIONS_CRM_DEALS
        int ACTIONS_EMAIL
        int USERS_CRM_CONTACTS
        int USERS_CRM_COMPANIES
        int USERS_CRM_DEALS
        int USERS_EMAIL
    }

    AGGREGATED_FEATURES {
        int id FK "References Company"
        float total_actions "Sum of all actions"
        float total_users "Sum of all users"
        float actions_per_user
        int days_active
        float activity_frequency
        float ACTIONS_CRM_CONTACTS_sum
        float ACTIONS_CRM_CONTACTS_mean
        float ACTIONS_CRM_CONTACTS_max
        float ACTIONS_CRM_CONTACTS_std
        float USERS_CRM_CONTACTS_sum
        float USERS_CRM_CONTACTS_mean
        float USERS_CRM_CONTACTS_max
    }
```

**Cardinality:**
- One Company → Many Usage Actions (1:N)
- One Company → One Aggregated Features (1:1)

**Data Sources:**
- `data/customers.csv` - 200 records
- `data/noncustomers.csv` - 5,003 records
- `data/usage_actions.csv` - 25,387 records

---

## 2. ML Pipeline Components ERD

### Core ML Components and Their Relationships

```mermaid
erDiagram
    CONFIG ||--|| TRAINER : "configures"
    TRAINER ||--|| DATA_LOADER : "uses"
    TRAINER ||--|| FEATURE_ENGINEER : "uses"
    TRAINER ||--|| MODEL : "trains"
    TRAINER ||--|| EVALUATOR : "uses"

    DATA_LOADER ||--|| SCHEMA_VALIDATOR : "validates with"
    DATA_LOADER ||--|| DATA_QUALITY_PROFILER : "profiles with"
    DATA_LOADER ||--|| DATA_DEDUPLICATOR : "deduplicates with"

    FEATURE_ENGINEER ||--|| EMPLOYEE_ENCODER : "uses"
    FEATURE_ENGINEER ||--|| INDUSTRY_ENCODER : "uses"
    FEATURE_ENGINEER ||--|| MISSING_VALUE_HANDLER : "uses"

    MODEL ||--o{ MODEL_IMPLEMENTATION : "implemented by"

    PREDICTOR ||--|| MODEL : "uses"
    PREDICTOR ||--|| FEATURE_ENGINEER : "uses"
    PREDICTOR ||--|| EMPLOYEE_ENCODER : "uses"
    PREDICTOR ||--|| INDUSTRY_ENCODER : "uses"
    PREDICTOR ||--|| MISSING_VALUE_HANDLER : "uses"

    API ||--|| PREDICTOR : "uses"

    TRAINED_ARTIFACT ||--|| MODEL : "contains"
    TRAINED_ARTIFACT ||--|| FEATURE_ENGINEER : "contains"
    TRAINED_ARTIFACT ||--|| EMPLOYEE_ENCODER : "contains"
    TRAINED_ARTIFACT ||--|| INDUSTRY_ENCODER : "contains"
    TRAINED_ARTIFACT ||--|| MISSING_VALUE_HANDLER : "contains"
    TRAINED_ARTIFACT ||--|| CONFIG : "contains"

    CONFIG {
        string experiment_name
        string experiment_description
        string mlflow_tracking_uri
        string data_customers_path
        string data_noncustomers_path
        string data_usage_path
        float test_size
        string model_type
        json hyperparameters
        int cv_folds
        float decision_threshold
        int random_seed
    }

    TRAINER {
        Config config
        HubSpotDataLoader data_loader
        BaseModel model
        FeatureEngineer feature_engineer
        Path artifact_dir
    }

    DATA_LOADER {
        string customers_path
        string noncustomers_path
        string usage_path
        int lookback_days
    }

    FEATURE_ENGINEER {
        string scaling_method "standard/minmax/robust/none"
        list numeric_features
        list categorical_features
        dict categorical_mappings
        bool is_fitted
        list feature_names
    }

    MODEL {
        bool _is_fitted
        string model_type
    }

    MODEL_IMPLEMENTATION {
        string type "LogisticRegression/RandomForest/XGBoost/LightGBM"
    }

    PREDICTOR {
        BaseModel model
        FeatureEngineer feature_engineer
        EmployeeRangeOrdinalEncoder employee_encoder
        IndustryEncoder industry_encoder
        MissingValueHandler missing_handler
    }

    API {
        string endpoint
        ModelService model_service
    }

    EVALUATOR {
        BaseModel model
        Config config
        Path artifact_dir
        array y_true
        array y_pred
        array y_proba
    }

    SCHEMA_VALIDATOR {
        list required_columns
        dict column_types
    }

    DATA_QUALITY_PROFILER {
        dict missing_values
        dict duplicates
    }

    DATA_DEDUPLICATOR {
        string strategy
    }

    EMPLOYEE_ENCODER {
        dict ordinal_mapping "Maps employee ranges to ordinal values (1-9)"
    }

    INDUSTRY_ENCODER {
        float min_frequency
        dict industry_mapping
    }

    MISSING_VALUE_HANDLER {
        string strategy "auto/median/mean/mode/zero/constant"
        dict impute_values
    }

    TRAINED_ARTIFACT {
        string artifact_dir
        datetime created_at
        string model_version
    }
```

---

## 3. Data Flow Diagram

### Training Pipeline Flow

```mermaid
flowchart TD
    A[Raw Data Sources] --> B[HubSpotDataLoader]
    A1[customers.csv] --> B
    A2[noncustomers.csv] --> B
    A3[usage_actions.csv] --> B

    B --> C[Schema Validation]
    B --> D[Data Quality Profiling]
    B --> E[Deduplication]

    C --> F[Combined Dataset]
    D --> F
    E --> F

    F --> G[Usage Aggregation]
    G --> H[Train/Test Split]

    H --> I[Preprocessing Pipeline]
    I --> I1[Employee Range Encoding]
    I1 --> I2[Industry Encoding]
    I2 --> I3[Missing Value Handling]
    I3 --> I4[Feature Scaling]

    I4 --> J[Model Training]
    J --> K{Model Type}

    K -->|LogisticRegression| L1[Logistic Regression]
    K -->|RandomForest| L2[Random Forest]
    K -->|XGBoost| L3[XGBoost]
    K -->|LightGBM| L4[LightGBM]

    L1 --> M[Trained Model]
    L2 --> M
    L3 --> M
    L4 --> M

    M --> N[Model Evaluation]
    N --> O[Metrics Calculation]
    N --> P[Visualization Generation]

    M --> Q[Artifact Saving]
    I --> Q
    O --> Q
    P --> Q

    Q --> R[Trained Artifact Directory]
    R --> S[MLflow Tracking]
```

### Inference Pipeline Flow

```mermaid
flowchart TD
    A[New Company Data] --> B{API or Direct?}

    B -->|API Request| C[FastAPI Endpoint]
    B -->|Direct| D[Predictor.predict]

    C --> D

    D --> E[Load Artifacts]
    E --> E1[Load Model]
    E --> E2[Load Preprocessors]

    E1 --> F[Apply Preprocessing]
    E2 --> F

    F --> F1[Employee Range Encoding]
    F1 --> F2[Industry Encoding]
    F2 --> F3[Missing Value Handling]
    F3 --> F4[Feature Scaling]

    F4 --> G[Model Prediction]
    G --> H[predict_proba]
    G --> I[predict]

    H --> J[Prediction Response]
    I --> J

    J --> K{Response Format}
    K -->|API| L[JSON Response]
    K -->|Direct| M[DataFrame/Array]
```

---

## 4. Component Hierarchy

### Class Inheritance and Composition

```mermaid
classDiagram
    class BaseModel {
        <<abstract>>
        +fit(X, y)
        +predict(X)
        +predict_proba(X)
        +get_feature_importance()
        -_is_fitted: bool
    }

    class LogisticRegressionModel {
        +model: LogisticRegression
        +fit(X, y)
        +predict(X)
    }

    class RandomForestModel {
        +model: RandomForestClassifier
        +fit(X, y)
        +predict(X)
    }

    class XGBoostModel {
        +model: XGBClassifier
        +fit(X, y)
        +predict(X)
    }

    class LightGBMModel {
        +model: LGBMClassifier
        +fit(X, y)
        +predict(X)
    }

    BaseModel <|-- LogisticRegressionModel
    BaseModel <|-- RandomForestModel
    BaseModel <|-- XGBoostModel
    BaseModel <|-- LightGBMModel

    class Trainer {
        -config: Config
        -data_loader: HubSpotDataLoader
        -model: BaseModel
        -feature_engineer: FeatureEngineer
        +train()
        +load_data()
        +preprocess_data()
    }

    class HubSpotDataLoader {
        -customers_path: str
        -validator: SchemaValidator
        -profiler: DataQualityProfiler
        +load()
        +validate()
    }

    class FeatureEngineer {
        -scaling_method: str
        -scaler: Scaler
        +fit(X)
        +transform(X)
        +fit_transform(X)
    }

    class Predictor {
        -model: BaseModel
        -feature_engineer: FeatureEngineer
        +predict(X)
        +predict_batch(X)
        +from_artifact_dir()
    }

    Trainer *-- HubSpotDataLoader
    Trainer *-- FeatureEngineer
    Trainer *-- BaseModel
    Predictor *-- BaseModel
    Predictor *-- FeatureEngineer
```

---

## 5. Artifact Storage Structure

```
artifacts/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── model.joblib                    # Trained model
    ├── feature_engineer.joblib         # Feature scaler
    ├── employee_encoder.joblib         # Employee range encoder
    ├── industry_encoder.joblib         # Industry encoder
    ├── missing_handler.joblib          # Missing value handler
    ├── config.yaml                     # Training configuration
    ├── metrics.json                    # Evaluation metrics
    ├── feature_importance.csv          # Feature importance scores
    ├── confusion_matrix.png            # Confusion matrix plot
    ├── roc_curve.png                   # ROC curve plot
    └── training_log.txt                # Training logs
```

---

## 6. API Endpoints and Data Models

### Request/Response Schema

```mermaid
erDiagram
    CompanyFeatures {
        int id
        float ALEXA_RANK
        string EMPLOYEE_RANGE
        string INDUSTRY
        float total_actions
        float total_users
        float actions_per_user
    }

    PredictionRequest {
        list~CompanyFeatures~ companies
    }

    PredictionResponse {
        int company_id
        int prediction "0 or 1"
        float conversion_probability
        string confidence "high/medium/low"
    }

    BatchPredictionResponse {
        list~PredictionResponse~ predictions
        int total_count
        datetime timestamp
    }

    PredictionRequest ||--o{ CompanyFeatures : "contains"
    BatchPredictionResponse ||--o{ PredictionResponse : "contains"
```

### API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | - | Welcome message |
| `/health` | GET | - | Health status |
| `/predict` | POST | `PredictionRequest` | `BatchPredictionResponse` |
| `/predict/single` | POST | `CompanyFeatures` | `PredictionResponse` |

---

## 7. Key Relationships Summary

### Data Entity Relationships
- **Company → UsageAction**: 1:N (One company generates many usage actions)
- **Company → AggregatedFeatures**: 1:1 (Each company has aggregated usage statistics)

### ML Component Relationships
- **Config → Trainer**: 1:1 (Config drives training process)
- **Trainer → Model**: 1:1 (Trains one model per run)
- **Trainer → FeatureEngineer**: 1:1 (Uses one feature engineering pipeline)
- **Predictor → Model**: 1:1 (Loads one trained model)
- **Predictor → Preprocessors**: 1:N (Uses multiple preprocessing components)
- **API → Predictor**: 1:1 (Wraps one predictor instance)

### Artifact Relationships
- **TrainedArtifact → Model**: 1:1 (Saved model)
- **TrainedArtifact → Preprocessors**: 1:N (Multiple preprocessing components)
- **TrainedArtifact → Config**: 1:1 (Training configuration)

---

## File References

- **Data Models**: `data/customers.csv`, `data/noncustomers.csv`, `data/usage_actions.csv`
- **Base Classes**: `src/ml_framework/models/base.py`, `src/ml_framework/data/base.py`
- **Implementations**: `src/ml_framework/models/implementations.py`
- **Data Loading**: `src/ml_framework/data/loaders.py`
- **Preprocessing**: `src/ml_framework/data/transformers.py`
- **Training**: `src/ml_framework/training/trainer.py`
- **Serving**: `src/ml_framework/serving/predictor.py`, `src/ml_framework/serving/api.py`
- **Configuration**: `configs/config.yaml`, `src/ml_framework/utils/config.py`
