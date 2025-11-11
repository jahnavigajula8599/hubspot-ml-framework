Machine Learning Models
=======================

The models module provides a flexible interface for different ML algorithms.

.. currentmodule:: ml_framework.models

BaseModel
---------

.. autoclass:: BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class defining the interface for all models.

   All model implementations must inherit from this class and implement:

   * :meth:`fit`: Train the model
   * :meth:`predict`: Make predictions
   * :meth:`predict_proba`: Predict class probabilities
   * :meth:`get_feature_importance`: Get feature importance scores

RandomForestModel
-----------------

.. autoclass:: RandomForestModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Random Forest classifier implementation using scikit-learn.

   **Hyperparameters:**

   * ``n_estimators`` (int): Number of trees in the forest
   * ``max_depth`` (int): Maximum depth of each tree
   * ``min_samples_split`` (int): Minimum samples required to split a node
   * ``random_state`` (int): Random seed for reproducibility

   **Example Usage:**

   .. code-block:: python

      from ml_framework.models import RandomForestModel

      model = RandomForestModel(config.model)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      probabilities = model.predict_proba(X_test)

      # Feature importance
      importance = model.get_feature_importance()
      print(importance)

XGBoostModel
------------

.. autoclass:: XGBoostModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   XGBoost classifier implementation.

   **Hyperparameters:**

   * ``n_estimators`` (int): Number of boosting rounds
   * ``max_depth`` (int): Maximum tree depth
   * ``learning_rate`` (float): Step size shrinkage
   * ``subsample`` (float): Subsample ratio of training instances

Model Selection Guide
---------------------

.. list-table:: Model Comparison
   :widths: 20 20 20 40
   :header-rows: 1

   * - Model
     - Training Speed
     - Accuracy
     - Use Case
   * - Random Forest
     - Baseline, interpretable
   * - XGBoost
     - Production, high accuracy
