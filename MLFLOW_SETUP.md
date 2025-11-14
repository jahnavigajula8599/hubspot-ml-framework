# MLflow UI Setup and Troubleshooting

## Issue Fixed

The MLflow UI was showing no experiments because:

1. **Missing Dependencies**: MLflow and seaborn were not installed
2. **No Experiments Created**: No training runs had been executed to create experiments

## Solution Applied

### 1. Installed Required Dependencies

```bash
pip install --user mlflow pandas scikit-learn joblib numpy matplotlib seaborn omegaconf pyyaml
```

### 2. Created Initial Experiment

Ran a training session to create the first experiment:

```python
from ml_framework.training import Trainer
from ml_framework.utils import load_config

config = load_config('configs/config.yaml')
trainer = Trainer(config)
results = trainer.train()
```

This created:
- **Experiment**: `customer_conversion_baseline`
- **Run**: `run_01` with metrics, model artifacts, and parameters
- **Model Registry**: Registered model `customer_conversion_baseline_model` version 1

## How to View Experiments in MLflow UI

### Option 1: Using the Convenience Script

```bash
./launch_mlflow_ui.sh
```

Or specify a custom port:

```bash
./launch_mlflow_ui.sh 8080
```

### Option 2: Using MLflow Command Directly

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open your browser to: **http://localhost:5000**

## What You'll See in the UI

Once launched, the MLflow UI will show:

1. **Experiments Tab**:
   - `customer_conversion_baseline` experiment
   - All training runs with their metrics and parameters

2. **Models Tab**:
   - `customer_conversion_baseline_model` (registered model)
   - Version 1 with lineage to the training run

3. **For Each Run**:
   - Metrics: accuracy, precision, recall, F1, ROC-AUC
   - Parameters: model hyperparameters, data settings
   - Artifacts: model files, plots, predictions
   - Tags: framework version, data version

## Running Additional Experiments

To create more experiments, simply run training with different configurations:

```python
# Modify configs/config.yaml first, then:
from ml_framework.training import Trainer
from ml_framework.utils import load_config

config = load_config('configs/config.yaml')
trainer = Trainer(config)
results = trainer.train()
```

Each training run will appear as a new entry in the MLflow UI.

## Verification

Check that experiments exist:

```bash
python -c "import mlflow; mlflow.set_tracking_uri('./mlruns'); exps = mlflow.search_experiments(); print(f'Found {len(exps)} experiments'); [print(f'  - {exp.name}') for exp in exps]"
```

Expected output:
```
Found 2 experiments:
  - customer_conversion_baseline
  - Default
```

## Troubleshooting

### "No experiments found"

**Cause**: No training has been run yet.

**Solution**: Run the training command above to create your first experiment.

### "ModuleNotFoundError: No module named 'mlflow'"

**Cause**: Dependencies not installed.

**Solution**:
```bash
pip install --user mlflow seaborn
```

### MLflow UI shows old/cached data

**Cause**: Browser cache.

**Solution**: Hard refresh your browser (Ctrl+Shift+R on most browsers).

### Port already in use

**Cause**: Another process is using port 5000.

**Solution**: Use a different port:
```bash
./launch_mlflow_ui.sh 8080
```

## Directory Structure

```
hubspot-ml-framework/
├── mlruns/                          # MLflow tracking directory
│   ├── 327303768480438271/         # Experiment ID
│   │   └── 8dda29230e6f.../        # Run ID
│   │       ├── artifacts/          # Model and plots
│   │       ├── metrics/            # Logged metrics
│   │       ├── params/             # Logged parameters
│   │       └── tags/               # Run tags
│   ├── models/                      # Model registry
│   └── 0/                           # Default experiment
└── artifacts/                       # Local artifacts (also logged to MLflow)
    └── customer_conversion_baseline_TIMESTAMP/
```

## Next Steps

1. Launch the MLflow UI: `./launch_mlflow_ui.sh`
2. Open http://localhost:5000 in your browser
3. Explore the experiment runs, metrics, and artifacts
4. Compare different runs to find the best model

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
