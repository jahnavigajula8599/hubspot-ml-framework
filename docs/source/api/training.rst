Model Training
==============

The training module orchestrates the ML workflow with MLflow integration.

.. currentmodule:: ml_framework.training

Trainer
-------

.. autoclass:: Trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Main training orchestrator that coordinates:

   * Data loading and preprocessing
   * Model training
   * Evaluation and metrics calculation
   * MLflow experiment tracking
   * Artifact saving

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      train
      _split_data
      _evaluate_model
      _save_artifacts
      _log_to_mlflow
      _generate_report

Training Workflow
-----------------

The complete training pipeline:

.. code-block:: text

   1. Load Data
      ├── company_data.csv
      └── user_activity.csv

   2. Preprocess
      ├── Merge datasets
      ├── Engineer features
      ├── Encode categorical
      └── Scale numerical

   3. Train Model
      ├── Split train/test (80/20)
      └── Fit model on training data

   4. Evaluate
      ├── Accuracy
      ├── Precision
      ├── Recall
      └── F1 Score

   5. Log to MLflow
      ├── Parameters
      ├── Metrics
      └── Artifacts

   6. Save Artifacts
      ├── model.pkl
      ├── preprocessor.pkl
      └── metadata.json

Example Usage
-------------

Basic Training
~~~~~~~~~~~~~~

.. code-block:: python

   from ml_framework.training import Trainer
   from ml_framework.utils import load_config

   # Load configuration
   config = load_config('configs/config.yaml')

   # Initialize and train
   trainer = Trainer(config)
   results = trainer.train()

   # Access results
   print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
   print(f"F1 Score: {results['metrics']['f1']:.4f}")
   print(f"Model saved to: {results['model_path']}")
   print(f"MLflow Run ID: {results['run_id']}")

With Custom Config
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from omegaconf import OmegaConf

   # Override configuration
   config = load_config('configs/config.yaml')
   config.model.hyperparameters.n_estimators = 200
   config.model.hyperparameters.max_depth = 15

   trainer = Trainer(config)
   results = trainer.train()

MLflow Integration
------------------

Viewing Experiments
~~~~~~~~~~~~~~~~~~~

Start the MLflow UI to view tracked experiments:

.. code-block:: bash

   mlflow ui --backend-store-uri ./mlruns
   # Open http://localhost:5000

What Gets Logged
~~~~~~~~~~~~~~~~

**Parameters:**

* Model type
* All hyperparameters
* Data split ratio
* Random seed

**Metrics:**

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC (if applicable)

**Artifacts:**

* Trained model (model.pkl)
* Preprocessor (preprocessor.pkl)
* Feature names
* Classification report
* Confusion matrix

Evaluation Metrics
------------------

.. automethod:: Trainer._evaluate_model

Metrics Explained
~~~~~~~~~~~~~~~~~

**Accuracy**: Overall correctness

.. math::

   \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}

**Precision**: Correctness of positive predictions

.. math::

   \\text{Precision} = \\frac{TP}{TP + FP}

**Recall**: Coverage of actual positives

.. math::

   \\text{Recall} = \\frac{TP}{TP + FN}

**F1 Score**: Harmonic mean of precision and recall

.. math::

   \\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}

Where:

* TP = True Positives
* TN = True Negatives
* FP = False Positives
* FN = False Negatives
