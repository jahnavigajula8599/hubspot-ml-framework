Data Processing
===============

The data module handles loading, validation, and preprocessing of data.

.. currentmodule:: ml_framework.data

DataLoader
----------

.. autoclass:: DataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      load_data
      merge_data
      _validate_data

   **Example Usage:**

   .. code-block:: python

      from ml_framework.data import DataLoader
      from ml_framework.utils import load_config

      config = load_config('configs/config.yaml')
      loader = DataLoader(config)
      company_df, activity_df = loader.load_data()
      merged_df = loader.merge_data(company_df, activity_df)

DataPreprocessor
----------------

.. autoclass:: DataPreprocessor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      preprocess
      _create_activity_features
      _encode_categorical
      _handle_missing
      _scale_features

   **Example Usage:**

   .. code-block:: python

      from ml_framework.data import DataPreprocessor

      preprocessor = DataPreprocessor(config)
      processed_df = preprocessor.preprocess(merged_df)

Feature Engineering
-------------------

The preprocessor creates the following features from user activity data:

* **total_actions**: Total number of user actions
* **total_users**: Count of unique users
* **days_active**: Number of days with activity
* **activity_frequency**: Average actions per day
* **user_diversity**: Ratio of unique users to total actions

Categorical Encoding
~~~~~~~~~~~~~~~~~~~~

Categorical variables are one-hot encoded:

* **EMPLOYEE_RANGE**: "1 to 10", "11 to 25", "26 to 50", etc.
* **INDUSTRY**: "COMPUTER_SOFTWARE", "INTERNET", "RETAIL", etc.

Numerical Scaling
~~~~~~~~~~~~~~~~~

Numerical features are scaled using MinMaxScaler to [0, 1] range:

* **ALEXA_RANK**: Website ranking
* **total_actions**: Aggregated activity count
* **activity_frequency**: Engagement metric
