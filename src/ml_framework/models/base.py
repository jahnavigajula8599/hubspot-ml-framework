"""
Base model protocol and abstract classes.

Defines the interface that all models must implement for consistency.
"""

from typing import Protocol, Any, Dict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class ModelProtocol(Protocol):
    """
    Protocol defining the interface for all ML models.
    
    Any model implementing these methods can be used in the framework,
    enabling flexibility to add new model types without framework changes.
    
    Design Rationale:
    - Consistent interface across all model types
    - Works with any scikit-learn compatible model
    - Easy to add custom models (neural networks, custom ensembles, etc.)
    - Testable: can mock models for unit tests
    
    Examples:
        >>> class MyCustomModel:
        ...     def fit(self, X, y):
        ...         # Training logic
        ...         return self
        ...     
        ...     def predict(self, X):
        ...         # Prediction logic
        ...         return predictions
        ...     
        ...     def predict_proba(self, X):
        ...         # Probability predictions
        ...         return probabilities
        >>> 
        >>> # Seamlessly integrates with framework
        >>> model = MyCustomModel()
        >>> trainer.train(model, X, y)
    """
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelProtocol':
        """
        Train the model on data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self for method chaining
        """
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        ...


class BaseModel(ABC):
    """
    Abstract base class for ML models with common functionality.
    
    Provides convenience methods and ensures consistent behavior
    across different model implementations.
    
    Examples:
        >>> class LogisticRegressionModel(BaseModel):
        ...     def __init__(self, **kwargs):
        ...         self.model = LogisticRegression(**kwargs)
        ...     
        ...     def fit(self, X, y):
        ...         self.model.fit(X, y)
        ...         self._is_fitted = True
        ...         return self
    """
    
    def __init__(self):
        """Initialize base model."""
        self._is_fitted = False
        self.model = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Default implementation delegates to underlying model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of class probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support predict_proba"
            )
        return self.model.predict_proba(X)
    
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance if available.
        
        Returns:
            Array of feature importances
            
        Raises:
            NotImplementedError: If model doesn't support feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support feature importance"
            )
    
    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(fitted={self._is_fitted})"
