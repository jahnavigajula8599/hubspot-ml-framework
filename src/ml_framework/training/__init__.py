"""
Training and evaluation modules.

Orchestrates model training, evaluation, and artifact management.
"""

from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    'Trainer',
    'Evaluator',
]
