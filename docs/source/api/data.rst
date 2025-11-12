Data Processing
===============

The data module handles loading, validation, and preprocessing of data.

.. currentmodule:: ml_framework.data

HubSpotDataLoader
------------------

.. autoclass:: HubSpotDataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      load_data
      load_and_prepare
      get_features_and_target
      _aggregate_usage_features
      _validate_customer_mrr

   **Example Usage:**

   .. code-block:: python

      from ml_framework.data import HubSpotDataLoader

      loader = HubSpotDataLoader(
          customers_path='data/customers.csv',
          noncustomers_path='data/noncustomers.csv',
          usage_path='data/usage.csv',
          lookback_days=30
      )
      df = loader.load_data()
      X, y = loader.get_features_and_target(df)

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
