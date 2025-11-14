# Entity Relationship Diagrams

This directory contains entity relationship diagrams (ERDs) for the HubSpot ML Framework.

## Available Documentation

### 1. Comprehensive ERD Document
- **File**: `../ENTITY_RELATIONSHIP_DIAGRAM.md`
- **Format**: Markdown with Mermaid diagrams
- **Content**:
  - Data Model ERD (Company, UsageAction, AggregatedFeatures)
  - ML Pipeline Components ERD
  - Data Flow Diagrams
  - Component Hierarchy
  - API Endpoints Schema

### 2. Visual Diagram Generator
- **File**: `../../scripts/generate_erd.py`
- **Purpose**: Generates PNG/SVG visual diagrams using Graphviz

## Viewing the Diagrams

### Option 1: View Mermaid Diagrams (Recommended)

The ERD document (`../ENTITY_RELATIONSHIP_DIAGRAM.md`) contains Mermaid diagrams that can be viewed:

1. **On GitHub**: Push to GitHub and view the markdown file directly
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Online Mermaid Editor**: Copy the Mermaid code to https://mermaid.live/
4. **Other markdown viewers**: Most modern markdown viewers support Mermaid

### Option 2: Generate Visual Diagrams

To generate PNG/SVG diagrams using the Python script:

#### Prerequisites

1. Install system Graphviz:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install graphviz

   # macOS
   brew install graphviz

   # Windows
   # Download from https://graphviz.org/download/
   ```

2. Install Python package:
   ```bash
   pip install graphviz
   ```

#### Generate Diagrams

Run the generator script:
```bash
cd /home/user/hubspot-ml-framework
python scripts/generate_erd.py
```

This will create the following files in this directory:
- `data_model_erd.png` - Data model visualization
- `data_model_erd.svg` - Data model (scalable)
- `ml_pipeline_erd.png` - ML pipeline components
- `ml_pipeline_erd.svg` - ML pipeline (scalable)
- `system_architecture.png` - Complete system architecture
- `system_architecture.svg` - System architecture (scalable)

## Diagram Types

### 1. Data Model ERD
Shows the core data entities and their relationships:
- **Company**: Core business entity with attributes like ALEXA_RANK, EMPLOYEE_RANGE, INDUSTRY
- **UsageAction**: Activity/behavior entity tracking user actions
- **AggregatedFeatures**: Derived features from usage aggregation

**Relationships**:
- Company → UsageAction (1:N)
- Company → AggregatedFeatures (1:1)

### 2. ML Pipeline ERD
Shows the machine learning pipeline components:
- **Config**: Configuration management
- **Trainer**: Orchestrates the training pipeline
- **DataLoader**: Loads and validates data
- **FeatureEngineer**: Handles feature transformations
- **Model**: Abstract base with multiple implementations (LogisticRegression, RandomForest, XGBoost, LightGBM)
- **Evaluator**: Evaluates model performance
- **Predictor**: Serves predictions
- **API**: FastAPI endpoints

### 3. System Architecture
Shows the complete end-to-end data flow:
- Data Sources → Loading → Preprocessing → Training → Artifacts → Serving

## Quick Reference

### Entity Relationships

**Data Entities**:
```
Company (1) ──── (N) UsageAction
   │
   │ (1:1)
   │
AggregatedFeatures
```

**ML Components**:
```
Config → Trainer → Model → Evaluator → Artifact
            │                            │
            ├─ DataLoader                ↓
            ├─ FeatureEngineer        Predictor → API
            └─ Preprocessors
```

### File Locations

| Component | File Path |
|-----------|-----------|
| Data files | `data/customers.csv`, `data/noncustomers.csv`, `data/usage_actions.csv` |
| Models | `src/ml_framework/models/` |
| Data loading | `src/ml_framework/data/loaders.py` |
| Transformers | `src/ml_framework/data/transformers.py` |
| Training | `src/ml_framework/training/trainer.py` |
| Serving | `src/ml_framework/serving/predictor.py`, `src/ml_framework/serving/api.py` |

## Additional Resources

- Main documentation: `../ENTITY_RELATIONSHIP_DIAGRAM.md`
- Project README: `../../README.md`
- Configuration: `../../configs/config.yaml`
- Generator script: `../../scripts/generate_erd.py`

## Notes

- The Mermaid diagrams provide interactive, version-control-friendly documentation
- The Python script generates high-quality static images for presentations
- Both formats are maintained for different use cases
- SVG format is recommended for scalability and embedding in documentation
