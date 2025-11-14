# Unit Tests Architecture

## Test Structure Overview

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration
└── test_data_validation.py        # All test classes (main test file)
    ├── TestBusinessLogic          # Business rule validation
    ├── TestPipelineBehavior       # Pipeline behavior verification
    └── TestDataIntegrity          # Data quality checks
```

---

## Test Class Breakdown

### 1. TestBusinessLogic
**Purpose:** Validate business rules on real data

```
TestBusinessLogic
├── Fixtures:
│   ├── customers_df           → loads data/customers.csv
│   ├── noncustomers_df        → loads data/noncustomers.csv
│   ├── usage_df               → loads data/usage_actions.csv
│   └── all_companies          → combines customers + noncustomers
│
└── Tests:
    ├── test_alexa_rank_reasonable_values()
    │   └── Validates: ALEXA_RANK > 0, <5% top-100 sites
    │
    ├── test_actions_must_exceed_users()
    │   └── Validates: ACTIONS >= USERS for all activity types
    │
    ├── test_usage_data_has_matching_companies()
    │   └── Validates: No orphan usage records
    │
    └── test_action_counts_non_negative()
        └── Validates: All action counts >= 0
```

**Data Flow:**
```
data/customers.csv ────┐
                       ├──→ all_companies → Business Rule Tests
data/noncustomers.csv ─┘

data/usage_actions.csv ──→ usage_df → Usage Validation Tests
```

---

### 2. TestPipelineBehavior
**Purpose:** Verify pipeline correctly handles data quality issues

```
TestPipelineBehavior
│
├── test_pipeline_removes_invalid_mrr_customers()
│   ├── Loads: Raw customers.csv
│   ├── Validates: Pipeline removes MRR <= 0
│   └── Uses: HubSpotDataLoader
│
├── test_pipeline_handles_duplicates()
│   ├── Loads: Raw customers + noncustomers
│   ├── Validates: No duplicates in final data
│   └── Uses: HubSpotDataLoader
│
└── test_pipeline_creates_required_features()
    ├── Loads: All data through pipeline
    ├── Validates: Required features exist
    │   ├── Usage aggregations (sum, mean)
    │   ├── Derived features (total_actions, days_active)
    │   └── Company attributes (ALEXA_RANK, INDUSTRY)
    └── Uses: HubSpotDataLoader.load_and_prepare()
```

**Pipeline Validation Flow:**
```
                    ┌─────────────────┐
                    │   Raw CSV Data  │
                    └────────┬────────┘
                             │
                             ↓
┌────────────────────────────────────────────────┐
│         HubSpotDataLoader Pipeline             │
│                                                │
│  1. Load raw data                             │
│  2. Remove MRR <= 0                           │
│  3. Deduplicate by ID                         │
│  4. Create features                           │
│  5. Validate schema                           │
└────────────────────┬───────────────────────────┘
                     │
                     ↓
        ┌────────────────────────┐
        │  Clean Training Data   │
        │  (X features, y target) │
        └────────────────────────┘
                     │
                     ↓
            ┌────────────────┐
            │  Test Validates:│
            │  ✓ No MRR <= 0  │
            │  ✓ No duplicates│
            │  ✓ All features │
            └────────────────┘
```

---

### 3. TestDataIntegrity
**Purpose:** Verify data quality and feature calculations

```
TestDataIntegrity
├── Fixtures:
│   └── usage_df → loads data/usage_actions.csv
│
└── Tests:
    ├── test_usage_has_timestamps()
    │   ├── Validates: WHEN_TIMESTAMP column exists
    │   └── Checks: <0.1% invalid timestamps
    │
    └── test_feature_calculations_on_real_data()
        ├── Validates: days_active calculation
        ├── Validates: activity_frequency calculation
        └── Tests on: First 5 companies (sample validation)
```

**Feature Calculation Validation:**
```
usage_actions.csv
     │
     ├─→ WHEN_TIMESTAMP ──→ days_active = max(timestamp) - min(timestamp)
     │                       └─→ Must be >= 0
     │
     └─→ ACTIONS_* columns ─→ total_actions / (days_active + 1)
                              └─→ activity_frequency >= 0
```

---

## Test Coverage Summary

| Test Class            | Tests | Purpose                          | Data Source           |
|-----------------------|-------|----------------------------------|-----------------------|
| TestBusinessLogic     | 4     | Business rule validation         | CSV files (raw)       |
| TestPipelineBehavior  | 3     | Pipeline behavior verification   | Via HubSpotDataLoader |
| TestDataIntegrity     | 2     | Data quality & calculations      | CSV files (raw)       |
| **TOTAL**             | **9** |                                  |                       |

---

## Business Rules Tested

### Data Quality Rules
1. **ALEXA_RANK**: Must be positive, <5% top-100 sites
2. **Actions >= Users**: Each user must perform ≥1 action
3. **No Orphan Data**: Every usage record must have matching company
4. **Non-negative Counts**: All action counts must be >= 0
5. **Valid MRR**: Customers must have MRR > 0
6. **No Duplicates**: Each company ID appears once
7. **Valid Timestamps**: <0.1% invalid timestamp tolerance

### Feature Engineering Rules
1. **days_active**: Must be >= 0
2. **activity_frequency**: Must be >= 0
3. **Required Features**: Pipeline must create:
   - Aggregations: `*_sum`, `*_mean`, `*_max`
   - Derived: `total_actions`, `total_users`, `days_active`
   - Attributes: `ALEXA_RANK`, `EMPLOYEE_RANGE`, `INDUSTRY`

---

## Running Tests

### Run All Tests
```bash
pytest tests/test_data_validation.py
```

### Run Specific Test Class
```bash
pytest tests/test_data_validation.py::TestBusinessLogic
pytest tests/test_data_validation.py::TestPipelineBehavior
pytest tests/test_data_validation.py::TestDataIntegrity
```

### Run Individual Test
```bash
pytest tests/test_data_validation.py::TestBusinessLogic::test_alexa_rank_reasonable_values -v
```

### Generate HTML Report
```bash
pytest tests/test_data_validation.py --html=test_report.html --self-contained-html
```

---

## Test Dependencies

```
conftest.py
    │
    └─→ Adds src/ to Python path
        └─→ Allows: from ml_framework.data import HubSpotDataLoader

test_data_validation.py
    │
    ├─→ pandas (data loading)
    ├─→ pytest (test framework)
    └─→ ml_framework.data.HubSpotDataLoader (pipeline tests only)
```

---

## What Each Test Validates

### Business Logic Layer
```
Raw Data → Business Rules → Valid/Invalid
                ↓
         Test Assertions
```

**Example:**
```python
# Test: ACTIONS >= USERS
raw_data["ACTIONS_CRM_CONTACTS"] = 5
raw_data["USERS_CRM_CONTACTS"] = 10
# ❌ FAIL: Violates business rule
```

### Pipeline Behavior Layer
```
Raw Data → Pipeline Processing → Clean Data
              ↓                      ↓
         Removes MRR<=0          Test Validates
         Deduplicates            No Bad Data
         Creates Features        All Features Present
```

**Example:**
```python
# Test: Pipeline removes MRR <= 0
raw_customers: 1000 rows (50 with MRR <= 0)
           ↓ HubSpotDataLoader
clean_data: 950 rows (0 with MRR <= 0)
# ✅ PASS: Pipeline removed invalid data
```

### Data Integrity Layer
```
Feature Calculations → Validate Math → Assert Correctness
```

**Example:**
```python
# Test: days_active calculation
timestamps = ["2024-01-01", "2024-01-15"]
days_active = (max - min).days = 14
assert days_active >= 0  # ✅ PASS
```

---

## Key Design Principles

1. **Test Real Data**: All tests run on actual CSV files, not mocked data
2. **Class-Scoped Fixtures**: Data loaded once per test class for efficiency
3. **Comprehensive Coverage**: Tests cover business logic, pipeline behavior, and data integrity
4. **Informative Output**: Print statements show data quality metrics during testing
5. **Single Test File**: All tests in one file (`test_data_validation.py`) for simplicity
