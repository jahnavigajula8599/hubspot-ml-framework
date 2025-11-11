Model Serving
=============

The serving module provides REST API endpoints for model predictions.

.. currentmodule:: ml_framework.serving

Predictor
---------

.. autoclass:: Predictor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Handles loading trained models and making predictions.

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      predict_single
      predict_batch
      _load_model
      _load_preprocessor
      _preprocess_input
      _calculate_confidence

   **Example Usage:**

   .. code-block:: python

      from ml_framework.serving import Predictor

      # Load trained model
      predictor = Predictor('artifacts/model_20231115_143000')

      # Single prediction
      company_data = {
          'id': 123,
          'ALEXA_RANK': 50000,
          'EMPLOYEE_RANGE': '26 to 50',
          'INDUSTRY': 'COMPUTER_SOFTWARE',
          'total_actions': 150,
          'total_users': 5,
          'days_active': 30,
          'activity_frequency': 5.0
      }

      result = predictor.predict_single(company_data)
      print(result)
      # {
      #   'company_id': 123,
      #   'prediction': 1,
      #   'conversion_probability': 0.78,
      #   'confidence': 'high'
      # }

FastAPI Application
-------------------

.. py:data:: app
   :type: fastapi.FastAPI

   Main FastAPI application instance.

API Endpoints
~~~~~~~~~~~~~

Health Check
^^^^^^^^^^^^

.. http:get:: /health

   Check if API is running and model is loaded.

   **Example Request:**

   .. code-block:: bash

      curl http://localhost:8000/health

   **Example Response:**

   .. code-block:: json

      {
        "status": "healthy",
        "model_loaded": true,
        "model_version": "1.0.0",
        "timestamp": "2023-11-15T14:30:00"
      }

Single Prediction
^^^^^^^^^^^^^^^^^

.. http:post:: /predict/single

   Make prediction for a single company.

   **Request Body:**

   .. code-block:: json

      {
        "id": 123,
        "ALEXA_RANK": 50000,
        "EMPLOYEE_RANGE": "26 to 50",
        "INDUSTRY": "COMPUTER_SOFTWARE",
        "total_actions": 150,
        "total_users": 5,
        "days_active": 30,
        "activity_frequency": 5.0
      }

   **Response:**

   .. code-block:: json

      {
        "company_id": 123,
        "prediction": 1,
        "conversion_probability": 0.78,
        "confidence": "high"
      }

   **Status Codes:**

   * ``200 OK``: Prediction successful
   * ``422 Unprocessable Entity``: Invalid input data
   * ``500 Internal Server Error``: Prediction failed

Batch Prediction
^^^^^^^^^^^^^^^^

.. http:post:: /predict/batch

   Make predictions for multiple companies.

   **Request Body:**

   .. code-block:: json

      [
        {"id": 101, "ALEXA_RANK": 10000, ...},
        {"id": 102, "ALEXA_RANK": 20000, ...},
        {"id": 103, "ALEXA_RANK": 30000, ...}
      ]

   **Response:**

   .. code-block:: json

      {
        "predictions": [
          {"company_id": 101, "prediction": 1, "conversion_probability": 0.85, "confidence": "high"},
          {"company_id": 102, "prediction": 0, "conversion_probability": 0.32, "confidence": "low"},
          {"company_id": 103, "prediction": 1, "conversion_probability": 0.67, "confidence": "medium"}
        ]
      }

Running the API
---------------

Starting the Server
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python run_api.py

The server will start on http://localhost:8000

Interactive Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI automatically generates interactive API documentation:

* **Swagger UI**: http://localhost:8000/docs
* **ReDoc**: http://localhost:8000/redoc

Testing from Python
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import requests

   # Health check
   response = requests.get('http://localhost:8000/health')
   print(response.json())

   # Single prediction
   company = {
       'id': 123,
       'ALEXA_RANK': 50000,
       'EMPLOYEE_RANGE': '26 to 50',
       'INDUSTRY': 'COMPUTER_SOFTWARE',
       'total_actions': 150,
       'total_users': 5,
       'days_active': 30,
       'activity_frequency': 5.0
   }

   response = requests.post(
       'http://localhost:8000/predict/single',
       json=company
   )
   print(response.json())

Pydantic Models
---------------

Input Validation
~~~~~~~~~~~~~~~~

.. py:class:: CompanyData

   Input schema for company data.

   :param int id: Company identifier
   :param Optional[int] ALEXA_RANK: Website ranking
   :param str EMPLOYEE_RANGE: Employee count range
   :param str INDUSTRY: Industry category
   :param int total_actions: Total user actions
   :param int total_users: Number of unique users
   :param int days_active: Days with activity
   :param float activity_frequency: Average actions per day

Output Schema
~~~~~~~~~~~~~

.. py:class:: PredictionResponse

   Output schema for prediction results.

   :param int company_id: Company identifier
   :param int prediction: 0 (non-customer) or 1 (customer)
   :param float conversion_probability: Probability of conversion [0, 1]
   :param str confidence: Confidence level (low/medium/high)

Confidence Levels
-----------------

Confidence is calculated based on probability:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Confidence
     - Probability Range
     - Interpretation
   * - High
     - < 0.3 or > 0.7
     - Model is very confident
   * - Medium
     - 0.4 - 0.6
     - Moderate confidence
   * - Low
     - 0.3 - 0.4 or 0.6 - 0.7
     - Lower confidence
