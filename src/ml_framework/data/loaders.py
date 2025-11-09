"""
Data loaders for HubSpot customer conversion prediction.

NOW WITH DATA QUALITY CHECKS!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# NEW IMPORTS
from .validators import SchemaValidator, DataQualityProfiler, DataDeduplicator

logger = logging.getLogger(__name__)


class HubSpotDataLoader:
    """
    Load and prepare HubSpot customer conversion data.
    
    NOW INCLUDES:
    - Schema validation
    - Data quality profiling  
    - Duplicate handling
    - Data quality reporting
    """
    COLUMN_DEFINITIONS = {
        "id": {
            "description": "Unique company identifier; also portal ID used in usage log.",
            "semantic_type": "entity_id",
            "dtype": "int",
            "source": ["customers", "noncustomers", "usage_actions"]
        },
        "CLOSEDATE": {
            "description": "Date when the company became a paying customer.",
            "semantic_type": "event_timestamp",
            "dtype": "datetime",
            "source": ["customers"]
        },
        "MRR": {
            "description": "Monthly Recurring Revenue at point of becoming a customer.",
            "semantic_type": "numeric",
            "dtype": "float",
            "source": ["customers"]
        },
        "ALEXA_RANK": {
            "description": "Web traffic ranking (lower = more traffic).",
            "semantic_type": "ordinal_numeric",
            "dtype": "int",
            "source": ["customers", "noncustomers"]
        },
        "EMPLOYEE_RANGE": {
            "description": "Employee count bucket; ranges like '11 to 25' or '10,001 or more'.",
            "semantic_type": "categorical_range",
            "dtype": "string",
            "source": ["customers", "noncustomers"]
        },
        "INDUSTRY": {
            "description": "Industry classification of company.",
            "semantic_type": "categorical",
            "dtype": "string",
            "source": ["customers", "noncustomers"]
        },
        "WHEN_TIMESTAMP": {
            "description": "Timestamp at which usage activity was logged.",
            "semantic_type": "event_timestamp",
            "dtype": "datetime",
            "source": ["usage_actions"]
        }
    }

    USAGE_METRIC_DEFINITIONS = {
        "ACTIONS_CRM_CONTACTS":  "Total number of CRM Contacts actions.",
        "ACTIONS_CRM_COMPANIES": "Total number of CRM Companies actions.",
        "ACTIONS_CRM_DEALS":     "Total number of CRM Deals actions.",
        "ACTIONS_EMAIL":         "Total number of Email actions.",
        "USERS_CRM_CONTACTS":    "Unique users interacting with Contacts.",
        "USERS_CRM_COMPANIES":   "Unique users interacting with Companies.",
        "USERS_CRM_DEALS":       "Unique users interacting with Deals.",
        "USERS_EMAIL":           "Unique users interacting with Email."
    }

    # You may combine them for convenience:
    COLUMN_DEFINITIONS.update(USAGE_METRIC_DEFINITIONS)

    def __init__(
        self,
        customers_path: str,
        noncustomers_path: str,
        usage_path: str,
        lookback_days: int = 30,
        validate_schema: bool = True,      # NEW!
        profile_data: bool = True,          # NEW!
        handle_duplicates: bool = True,     # NEW!
        duplicate_strategy: str = 'most_complete'  # NEW!
    ):
        """
        Initialize data loader with quality checks.
        
        Args:
            ... (existing args)
            validate_schema: Whether to validate schemas
            profile_data: Whether to profile data quality
            handle_duplicates: Whether to remove duplicates
            duplicate_strategy: How to handle duplicates
        """
        self.customers_path = Path(customers_path)
        self.noncustomers_path = Path(noncustomers_path)
        self.usage_path = Path(usage_path)
        self.lookback_days = lookback_days
        
        # NEW: Data quality components
        self.validate_schema = validate_schema
        self.profile_data = profile_data
        self.handle_duplicates = handle_duplicates
        self.duplicate_strategy = duplicate_strategy
        
        # Initialize validators
        if self.validate_schema:
            self.validator = SchemaValidator()
        if self.profile_data:
            self.profiler = DataQualityProfiler()
        if self.handle_duplicates:
            self.deduplicator = DataDeduplicator(key_column='id')
        
        # Validate paths
        for path in [self.customers_path, self.noncustomers_path, self.usage_path]:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge all data sources WITH QUALITY CHECKS.
        
        Returns:
            DataFrame with company features and target variable
        """
        logger.info("="*60)
        logger.info("LOADING DATA WITH QUALITY CHECKS")
        logger.info("="*60)
        
        # Load individual datasets
        customers = pd.read_csv(self.customers_path)
        noncustomers = pd.read_csv(self.noncustomers_path)
        usage = pd.read_csv(self.usage_path)
        
        logger.info(f"Loaded {len(customers)} customers")
        logger.info(f"Loaded {len(noncustomers)} non-customers")
        logger.info(f"Loaded {len(usage)} usage records")
        
        # Validate schemas
        if self.validate_schema:
            logger.info("\n🔍 Validating schemas...")
            self.validator.validate(customers, 'customers')
            self.validator.validate(noncustomers, 'noncustomers')
            self.validator.validate(usage, 'usage_actions')
        
        # Profile data quality
        if self.profile_data:
            logger.info("\n📊 Profiling data quality...")
            self.customers_profile = self.profiler.profile(customers, 'customers')
            self.noncustomers_profile = self.profiler.profile(noncustomers, 'noncustomers')
            self.usage_profile = self.profiler.profile(usage, 'usage_actions')
        
        # Handle duplicates
        if self.handle_duplicates:
            logger.info("\n🔧 Handling duplicates...")
            customers = self.deduplicator.deduplicate(
                customers, 
                strategy=self.duplicate_strategy,
                save_duplicates=True,
                df_name='customers' 
            )
            noncustomers = self.deduplicator.deduplicate(
                noncustomers, 
                strategy=self.duplicate_strategy,
                save_duplicates=True,
                df_name='noncustomers'  
            )
        # Validate and clean MRR business rules
        logger.info("\n🎯 Validating customer MRR...")
        customers, removed_count = self._validate_customer_mrr(customers)
               
        # Add target variable
        customers['is_customer'] = 1
        noncustomers['is_customer'] = 0
        
        # Combine customers and non-customers
        companies = pd.concat([customers, noncustomers], ignore_index=True)
        
        logger.info(f"\n✅ Total companies: {len(companies)}")
        
        # Aggregate usage features
        usage_features = self._aggregate_usage_features(usage)
        
        # Merge with company data
        df = companies.merge(usage_features, on='id', how='left')
        
        # Fill missing usage features with 0 (companies with no usage)
        usage_cols = usage_features.columns.drop('id')
        df[usage_cols] = df[usage_cols].fillna(0)
        
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Features: {df.shape[1] - 1}, Target: is_customer")
        logger.info("="*60 + "\n")
        
        return df
    
    def _aggregate_usage_features(self, usage: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate usage actions into meaningful features.
        
        Args:
            usage: Raw usage actions dataframe
            
        Returns:
            DataFrame with aggregated usage features per company
        """
        logger.info("Aggregating usage features...")
        
        # Convert timestamp to datetime
        usage['WHEN_TIMESTAMP'] = pd.to_datetime(usage['WHEN_TIMESTAMP'])
        
        # Filter to lookback window (if needed for production)
        # For this assessment, we use all available data
        # cutoff_date = usage['WHEN_TIMESTAMP'].max() - pd.Timedelta(days=self.lookback_days)
        # usage = usage[usage['WHEN_TIMESTAMP'] >= cutoff_date]
        
        # Define aggregation functions
        agg_dict = {
            # Total actions across all objects
            'ACTIONS_CRM_CONTACTS': ['sum', 'mean', 'max', 'std'],
            'ACTIONS_CRM_COMPANIES': ['sum', 'mean', 'max', 'std'],
            'ACTIONS_CRM_DEALS': ['sum', 'mean', 'max', 'std'],
            'ACTIONS_EMAIL': ['sum', 'mean', 'max', 'std'],
            
            # Unique users (engagement)
            'USERS_CRM_CONTACTS': ['sum', 'mean', 'max'],
            'USERS_CRM_COMPANIES': ['sum', 'mean', 'max'],
            'USERS_CRM_DEALS': ['sum', 'mean', 'max'],
            'USERS_EMAIL': ['sum', 'mean', 'max'],
            
            # Temporal features
            'WHEN_TIMESTAMP': ['count', 'min', 'max']
        }
        
        # Group by company ID and aggregate
        usage_agg = usage.groupby('id').agg(agg_dict)
        
        # Flatten multi-level columns
        usage_agg.columns = [
            f"{col[0]}_{col[1]}" for col in usage_agg.columns
        ]
        
        # Create additional derived features
        usage_agg['total_actions'] = (
            usage_agg['ACTIONS_CRM_CONTACTS_sum'] +
            usage_agg['ACTIONS_CRM_COMPANIES_sum'] +
            usage_agg['ACTIONS_CRM_DEALS_sum'] +
            usage_agg['ACTIONS_EMAIL_sum']
        )
        
        usage_agg['total_users'] = (
            usage_agg['USERS_CRM_CONTACTS_sum'] +
            usage_agg['USERS_CRM_COMPANIES_sum'] +
            usage_agg['USERS_CRM_DEALS_sum'] +
            usage_agg['USERS_EMAIL_sum']
        )
        
        # Actions per user (engagement intensity)
        usage_agg['actions_per_user'] = (
            usage_agg['total_actions'] / usage_agg['total_users']
        ).fillna(0)
        
        # Temporal features
        usage_agg['days_active'] = (
            usage_agg['WHEN_TIMESTAMP_max'] - usage_agg['WHEN_TIMESTAMP_min']
        ).dt.days
        
        usage_agg['activity_frequency'] = (
            usage_agg['WHEN_TIMESTAMP_count'] / (usage_agg['days_active'] + 1)
        )
        
        # Drop raw timestamp columns
        usage_agg = usage_agg.drop(
            ['WHEN_TIMESTAMP_count', 'WHEN_TIMESTAMP_min', 'WHEN_TIMESTAMP_max'],
            axis=1
        )
        
        # Reset index to make ID a column
        usage_agg = usage_agg.reset_index()
        
        # Fill any remaining NaN with 0
        usage_agg = usage_agg.fillna(0)
        
        # Replace inf values with 0
        usage_agg = usage_agg.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Created {len(usage_agg.columns) - 1} usage features")
        
        return usage_agg
    
    def get_features_and_target(
        self,
        df: pd.DataFrame,
        drop_columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target.
        
        Args:
            df: Input dataframe
            drop_columns: Additional columns to drop
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Columns to drop (non-feature columns)
        default_drop_cols = ['id', 'is_customer', 'CLOSEDATE', 'MRR']
        
        if drop_columns:
            default_drop_cols.extend(drop_columns)
        
        # Remove duplicates
        default_drop_cols = list(set(default_drop_cols))
        
        # Keep only columns that exist
        cols_to_drop = [col for col in default_drop_cols if col in df.columns]
        
        # Extract target
        y = df['is_customer']
        
        # Extract features
        X = df.drop(cols_to_drop, axis=1)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"Class balance: {y.mean():.2%} customers")
        
        return X, y
    
    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data and return features and target.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.load_data()
        return self.get_features_and_target(df)
    
    def _validate_customer_mrr(self, customers: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Validate that customers have valid MRR values.
        
        Business Rule: is_customer = 1 → MRR must be > 0
        
        Args:
            customers: Customer dataframe
            
        Returns:
            Tuple of (clean_customers, removed_count)
        """
        original_count = len(customers)
        
        # Check for invalid MRR
        invalid_mask = (customers['MRR'] <= 0)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            logger.warning(
                f"\n{'='*70}\n"
                f"⚠️  Found {invalid_count} customers with MRR <= 0 ({invalid_count/original_count*100:.2f}%)\n"
                f"{'='*70}\n"
                f"Business rule: Customers (is_customer = 1) must have MRR > 0\n"
            )
            
            # Show sample
            invalid_sample = customers[invalid_mask][['id', 'MRR']].head(5)
            logger.warning(f"Sample invalid rows:\n{invalid_sample}\n")
            
            # Save removed rows
            removed = customers[invalid_mask].copy()
            output_dir = Path('artifacts/data_quality')
            output_dir.mkdir(parents=True, exist_ok=True)
            removed.to_csv(output_dir / 'removed_invalid_customers.csv', index=False)
            logger.info(f"💾 Saved removed rows to: artifacts/data_quality/removed_invalid_customers.csv")
            
            # Remove invalid rows
            customers_clean = customers[~invalid_mask].copy()
            logger.warning(f"🗑️  Removed {invalid_count} invalid customers")
            logger.info(f"✓ Retained {len(customers_clean)} valid customers ({len(customers_clean)/original_count*100:.2f}%)")
            logger.warning(f"{'='*70}")
            
            return customers_clean, invalid_count
        else:
            logger.info(f"✓ All {original_count} customers have valid MRR > 0")
            return customers, 0