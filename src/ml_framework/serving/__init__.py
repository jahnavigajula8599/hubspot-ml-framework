"""
Model serving and prediction modules.

Provides utilities for deploying and using trained models.
"""

from .predictor import Predictor
from .api import app

__all__ = [
    'Predictor', 'app'
]
