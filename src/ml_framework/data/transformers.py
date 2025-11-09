"""
Feature transformation and preprocessing utilities.

Provides composable feature transformers for data preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Main feature engineering pipeline.
    
    Handles categorical encoding, feature scaling, and feature selection.
    
    Design Rationale:
    - Fits on training data, transforms both train and test
    - Stores fitted scalers/encoders for consistent test-time transformation
    - Handles missing values gracefully
    
    Examples:
        >>> engineer = FeatureEngineer(scaling_method='standard')
        >>> X_train_transformed = engineer.fit_transform(X_train)
        >>> X_test_transformed = engineer.transform(X_test)
    """
    
    def __init__(
        self,
        scaling_method: str = 'standard',
        handle_categorical: bool = True,
        exclude_from_scaling: Optional[List[str]] = None
    ):
        """
        Initialize feature engineer.
        
        Args:
            scaling_method: Method for scaling ('standard', 'minmax', 'robust', 'none')
            handle_categorical: Whether to encode categorical features
            exclude_from_scaling: List of column names to exclude from scaling
        """
        self.scaling_method = scaling_method
        self.handle_categorical = handle_categorical
        self.exclude_from_scaling = exclude_from_scaling or []
        
        # Initialize scaler based on method
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        # Store fitted state
        self.is_fitted = False
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        self.categorical_mappings = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Args:
            X: Training features
            y: Optional training target
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature engineer...")
        
        X = X.copy()
        self.feature_names = X.columns.tolist()
        
        # Identify numeric and categorical features
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        logger.info(f"Found {len(self.numeric_features)} numeric features")
        logger.info(f"Found {len(self.categorical_features)} categorical features")
        
        # Handle categorical features
        if self.handle_categorical and self.categorical_features:
            for col in self.categorical_features:
                # Create mapping for each category
                unique_vals = X[col].dropna().unique()
                self.categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
                logger.info(f"Encoded {col}: {len(unique_vals)} unique values")
        
        # Fit scaler on numeric features
        if self.scaler is not None:
            # Exclude specified columns
            scale_features = [
                col for col in self.numeric_features
                if col not in self.exclude_from_scaling
            ]
            
            if scale_features:
                self.scaler.fit(X[scale_features])
                logger.info(f"Fitted scaler on {len(scale_features)} features")
        
        self.is_fitted = True
        logger.info("Feature engineer fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted parameters.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X = X.copy()
        
        # Handle categorical features
        if self.handle_categorical and self.categorical_features:
            for col in self.categorical_features:
                if col in X.columns:
                    # Map using stored mappings, unknown values get -1
                    X[col] = X[col].map(self.categorical_mappings[col]).fillna(-1)
        
        # Scale numeric features
        if self.scaler is not None:
            scale_features = [
                col for col in self.numeric_features
                if col not in self.exclude_from_scaling and col in X.columns
            ]
            
            if scale_features:
                X[scale_features] = self.scaler.transform(X[scale_features])
        
        return X
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit on training data and transform.
        
        Args:
            X: Training features
            y: Optional training target
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)


class EmployeeRangeOrdinalEncoder:
    """
    Ordinal encoder for EMPLOYEE_RANGE.

    Converts ranges such as:
        '1'
        '2 to 5'
        '6 to 10'
        '51 to 200'
        '10,001 or more'
        nan

    Into an ordinal scale based on company size hierarchy.
    """

    def __init__(self):
        # Define explicit ordering
        self.ordering = {
            "1": 1,
            "2 to 5": 2,
            "6 to 10": 3,
            "11 to 25": 4,
            "26 to 50": 5,
            "51 to 200": 6,
            "201 to 1000": 7,
            "1001 to 10000": 8,
            "10,001 or more": 9
        }

        self.missing_value = -1   # keep missing separate

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "EMPLOYEE_RANGE" not in df.columns:
            return df

        encoded_values = []

        for val in df["EMPLOYEE_RANGE"]:
            if pd.isna(val):
                encoded_values.append(self.missing_value)
            else:
                val_str = str(val).strip()
                encoded_values.append(self.ordering.get(val_str, self.missing_value))

        df["employee_range_ordinal"] = encoded_values

        # Drop original column
        df = df.drop("EMPLOYEE_RANGE", axis=1)

        return df


class IndustryEncoder:
    """
    Custom transformer for INDUSTRY feature.
    
    One-hot encodes industry while handling rare categories.
    
    Examples:
        >>> encoder = IndustryEncoder(min_frequency=10)
        >>> encoder.fit(X_train)
        >>> X_train = encoder.transform(X_train)
    """
    
    def __init__(self, min_frequency: int = 5):
        """
        Initialize industry encoder.
        
        Args:
            min_frequency: Minimum frequency for a category to get its own column
        """
        self.min_frequency = min_frequency
        self.top_industries = []
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'IndustryEncoder':
        """
        Fit encoder on training data.
        
        Args:
            X: Training features
            
        Returns:
            Self for method chaining
        """
        if 'INDUSTRY' not in X.columns:
            self.is_fitted = True
            return self
        
        # Find top industries
        industry_counts = X['INDUSTRY'].value_counts()
        self.top_industries = industry_counts[
            industry_counts >= self.min_frequency
        ].index.tolist()
        
        logger.info(
            f"Industry encoder: {len(self.top_industries)} categories "
            f"(min_frequency={self.min_frequency})"
        )
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform industry column.
        
        Args:
            X: Features to transform
            
        Returns:
            DataFrame with industry encoded
        """
        if not self.is_fitted:
            raise ValueError("IndustryEncoder must be fitted before transform")
        
        X = X.copy()
        
        if 'INDUSTRY' not in X.columns:
            return X
        
        # Handle NaN explicitly - map to 'Other'
        X['INDUSTRY'] = X['INDUSTRY'].fillna('Other')
        
        # Replace rare industries with 'Other'
        X['INDUSTRY'] = X['INDUSTRY'].apply(
            lambda x: x if x in self.top_industries else 'Other'
        )
        
        # One-hot encode
        industry_dummies = pd.get_dummies(X['INDUSTRY'], prefix='industry')
        
        # Ensure all expected columns exist (for test set)
        for industry in self.top_industries + ['Other']:
            col_name = f'industry_{industry}'
            if col_name not in industry_dummies.columns:
                industry_dummies[col_name] = 0
        
        # Drop original column and concat dummies
        X = X.drop('INDUSTRY', axis=1)
        X = pd.concat([X, industry_dummies], axis=1)
        
        return X
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class MissingValueHandler:
    """
    Handle missing values in dataset.
    
    Examples:
        >>> handler = MissingValueHandler(strategy='auto')
        >>> X = handler.fit_transform(X_train)
    """
    
    def __init__(self, strategy: str = 'auto', constant_value: int = -1):
        """
        Initialize missing value handler.
        
        Args:
            strategy: Strategy for imputation 
                     'auto' - numeric→median, categorical→mode
                     'median' - fill all with median
                     'mode' - fill all with mode
                     'mean' - fill all with mean
                     'zero' - fill all with 0
                     'constant' - fill all with constant_value
            constant_value: Value to use when strategy='constant'
        """
        self.strategy = strategy
        self.constant_value = constant_value
        self.fill_values = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        """
        Fit imputation strategy on training data.
        
        Args:
            X: Training features
            
        Returns:
            Self for method chaining
        """
        # Identify numeric vs categorical columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        for col in X.columns:
            if X[col].isna().any():
                # Auto strategy: numeric→median, categorical→mode
                if self.strategy == 'auto':
                    if col in numeric_cols:
                        self.fill_values[col] = X[col].median()
                    else:
                        mode_val = X[col].mode()
                        self.fill_values[col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                
                # Specific strategies
                elif self.strategy == 'mean':
                    self.fill_values[col] = X[col].mean()
                elif self.strategy == 'median':
                    self.fill_values[col] = X[col].median()
                elif self.strategy == 'mode':
                    mode_val = X[col].mode()
                    self.fill_values[col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                elif self.strategy == 'zero':
                    self.fill_values[col] = 0
                elif self.strategy == 'constant':
                    self.fill_values[col] = self.constant_value
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if self.fill_values:
            logger.info(f"Will impute {len(self.fill_values)} columns with strategy='{self.strategy}'")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values.
        
        Args:
            X: Features to transform
            
        Returns:
            DataFrame with imputed values
        """
        if not self.is_fitted:
            raise ValueError("MissingValueHandler must be fitted before transform")
        
        X = X.copy()
        
        for col, fill_value in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
