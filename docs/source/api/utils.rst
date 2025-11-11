Utilities
=========

Helper functions and utilities for configuration, logging, and more.

.. currentmodule:: ml_framework.utils

Configuration
-------------

.. autofunction:: load_config

.. autofunction:: validate_config

.. autofunction:: merge_configs

**Example Usage:**

.. code-block:: python

   from ml_framework.utils import load_config

   # Load configuration
   config = load_config('configs/config.yaml')

   # Access values
   print(config.experiment.name)
   print(config.model.type)
   print(config.model.hyperparameters.n_estimators)

   # Override values
   config.model.hyperparameters.max_depth = 15

Configuration Structure
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   experiment:
     name: "hubspot_customer_conversion"
     mlflow_tracking_uri: "file:./mlruns"

   data:
     company_data_path: "data/company_data.csv"
     activity_data_path: "data/user_activity.csv"
     test_size: 0.2

   model:
     type: "random_forest"
     hyperparameters:
       n_estimators: 100
       max_depth: 10
       min_samples_split: 5

   features:
     categorical:
       - EMPLOYEE_RANGE
       - INDUSTRY
     numerical:
       - ALEXA_RANK
       - total_actions
       - total_users
       - activity_frequency

   reproducibility:
     seed: 42

Logging
-------

.. autofunction:: setup_logging

.. autofunction:: get_logger

.. autoclass:: ColoredFormatter
   :members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

   from ml_framework.utils import setup_logging, get_logger

   # Setup logging
   setup_logging(level='INFO', log_file='logs/app.log')

   # Get logger for your module
   logger = get_logger(__name__)

   # Use logger
   logger.info('Training started')
   logger.warning('Low memory available')
   logger.error('Failed to load data')

Logging Levels
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Level
     - Purpose
   * - DEBUG
     - Detailed information for debugging
   * - INFO
     - General informational messages
   * - WARNING
     - Warning messages for potential issues
   * - ERROR
     - Error messages for failures
   * - CRITICAL
     - Critical errors that may cause shutdown
