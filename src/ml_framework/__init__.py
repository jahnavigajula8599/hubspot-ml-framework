"""
ML Framework for Customer Conversion Prediction.

A production-grade, extensible framework for building and deploying
machine learning models.

Quick Start:
    >>> from ml_framework.training import Trainer
    >>> from ml_framework.utils import load_config
    >>>
    >>> config = load_config('configs/config.yaml')
    >>> trainer = Trainer(config)
    >>> results = trainer.train()
"""

__version__ = "0.1.0"

from . import data, models, serving, training, utils

__all__ = [
    "data",
    "models",
    "training",
    "serving",
    "utils",
]
