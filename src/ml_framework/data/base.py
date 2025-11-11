"""
Base data loader protocol.

Defines the interface that all data loaders must implement.
Using Protocol allows for duck typing and loose coupling.
"""

from typing import Protocol, Tuple

import pandas as pd


class DataLoader(Protocol):
    """
    Protocol defining the interface for data loaders.

    Any class implementing these methods can be used as a DataLoader,
    enabling easy extension to new data sources (databases, APIs, etc.)
    without modifying existing code.

    Design Rationale:
        - Protocol-based design allows for flexible implementations
        - Loose coupling: components depend on interface, not concrete classes
        - Easy to test: can create mock loaders for unit tests
        - Extensible: add new data sources without changing framework code

    Examples:
        >>> class MyCustomLoader:
        ...     def load_data(self):
        ...         # Custom loading logic
        ...         return pd.DataFrame(...)
        ...
        ...     def get_features_and_target(self, df):
        ...         X = df.drop('target', axis=1)
        ...         y = df['target']
        ...         return X, y
        >>>
        >>> # Works seamlessly with the framework
        >>> loader = MyCustomLoader()
        >>> X, y = loader.load_and_prepare()
    """

    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from source.

        Returns:
            DataFrame with raw data
        """
        ...

    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features (X) and target (y).

        Args:
            df: Input dataframe

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        ...

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convenience method to load data and split into X, y.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.load_data()
        return self.get_features_and_target(df)


class FeatureTransformer(Protocol):
    """
    Protocol for feature transformation components.

    Allows creating modular, composable feature engineering pipelines.

    Examples:
        >>> class LogTransformer:
        ...     def transform(self, df):
        ...         df['log_revenue'] = np.log1p(df['revenue'])
        ...         return df
        >>>
        >>> class RecencyTransformer:
        ...     def transform(self, df):
        ...         df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
        ...         return df
        >>>
        >>> # Compose transformers
        >>> pipeline = [LogTransformer(), RecencyTransformer()]
        >>> for transformer in pipeline:
        ...     df = transformer.transform(df)
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by adding/modifying features.

        Args:
            df: Input dataframe

        Returns:
            Transformed dataframe
        """
        ...

    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit transformer on data and transform.

        Optional method for stateful transformers (e.g., scalers).

        Args:
            df: Input dataframe
            y: Optional target variable

        Returns:
            Transformed dataframe
        """
        return self.transform(df)
