.. HubSpot ML Framework documentation master file, created by
   sphinx-quickstart on Mon Nov 10 09:18:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HubSpot ML Framework Documentation
===================================

Production-grade machine learning framework for customer conversion prediction
with MLflow experiment tracking and FastAPI serving.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/mlflow-2.8+-green.svg
   :alt: MLflow Version

.. image:: https://img.shields.io/badge/fastapi-0.104+-red.svg
   :alt: FastAPI Version

Features
--------

* **Experiment Tracking**: Complete MLflow integration for reproducible research
* **Production Serving**: FastAPI REST API with automatic validation
* **Modular Design**: Easily extend with new models and features
* **Config-Driven**: YAML configuration for all experiments
* **Multiple Models**: Support for Logistic Regression, Random Forest, XGBoost, and more
* **Type-Safe**: Full type hints throughout the codebase

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/jahnavigajula8599/hubspot-ml-framework
   cd hubspot-ml-framework
   pip install -r requirements.txt

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ml_framework.training import Trainer
   from ml_framework.utils import load_config

   config = load_config('configs/config.yaml')
   trainer = Trainer(config)
   results = trainer.train()

   print(f"Accuracy: {results['metrics']['accuracy']:.2f}")

Serving Predictions
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start MLflow UI
   mlflow ui --backend-store-uri ./mlruns

   # Start FastAPI server
   python run_api.py

   # Make predictions
   curl -X POST http://localhost:8000/predict/single \\
     -H "Content-Type: application/json" \\
     -d '{"id": 123, "ALEXA_RANK": 50000, ...}'

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   training
   serving
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/models
   api/training
   api/serving
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   architecture
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
