"""
Data validation utilities for ensuring data quality.

This module provides:
- Schema validation (required/optional columns, dtypes, composite keys)
- Soft vs. hard validation modes
- Data quality profiling (missingness, duplicates, numeric summaries)
- Intelligent deduplication utilities

"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path 
logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validate dataframe schemas against expected structure.

    Enhancements in this rewrite:
    - Composite unique key support (('id', 'WHEN_TIMESTAMP'))
    - Optional column type validation (only if types defined)
    - Soft vs. hard errors (prospects can legitimately have duplicates)
    - Clearer warnings and structured logging
    """

    # EXPECTED SCHEMA based on assessment definitions
    EXPECTED_SCHEMAS = {
        "customers": {
            "required_columns": ["id", "CLOSEDATE", "MRR"],
            "optional_columns": ["ALEXA_RANK", "EMPLOYEE_RANGE", "INDUSTRY"],
            "types": {
                "id": ["int64", "int32"],
                "CLOSEDATE": ["object", "datetime64[ns]"],
                "MRR": ["float64", "float32", "int64"],  # MRR may load as int if no decimals
            },
            # Hard uniqueness (true customers have exactly one row per id)
            "unique_keys": [("id",)],
            "soft_unique_keys": []  # No soft keys for customers
        },

        "noncustomers": {
            "required_columns": ["id"],
            "optional_columns": ["ALEXA_RANK", "EMPLOYEE_RANGE", "INDUSTRY"],
            "types": {
                "id": ["int64", "int32"],
            },
            # Prospects may have duplicated ids due to ingestion → treat as soft validation
            "unique_keys": [],
            "soft_unique_keys": [("id",)]  # Warn instead of error
        },

        "usage_actions": {
            "required_columns": ["id", "WHEN_TIMESTAMP"],
            "optional_columns": [
                "ACTIONS_CRM_CONTACTS", "ACTIONS_CRM_COMPANIES",
                "ACTIONS_CRM_DEALS", "ACTIONS_EMAIL",
                "USERS_CRM_CONTACTS", "USERS_CRM_COMPANIES",
                "USERS_CRM_DEALS", "USERS_EMAIL"
            ],
            "types": {
                "id": ["int64", "int32"],
                "WHEN_TIMESTAMP": ["object", "datetime64[ns]"],
            },
            # Soft composite key validation → warn if duplicates exist
            "unique_keys": [],
            "soft_unique_keys": [("id", "WHEN_TIMESTAMP")]
        }
    }

    def validate(self, df: pd.DataFrame, schema_name: str) -> bool:
        """
        Validate dataframe against the expected schema.

        Rules:
        - Required columns → hard error if missing.
        - Unexpected columns → warning only.
        - Type mismatches → warning only.
        - Unique keys:
             unique_keys → HARD validation
             soft_unique_keys → WARNING-only

        Returns
        -------
        True if validation passes without *hard* failures.
        """

        if schema_name not in self.EXPECTED_SCHEMAS:
            raise ValueError(f"Unknown schema: {schema_name}")

        schema = self.EXPECTED_SCHEMAS[schema_name]
        errors = []
        warnings = []

        # --------------------------------------------
        # 1. Required columns must exist
        # --------------------------------------------
        missing_required = set(schema["required_columns"]) - set(df.columns)
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")

        # --------------------------------------------
        # 2. Warn on unexpected extra columns
        # --------------------------------------------
        expected_columns = set(schema["required_columns"] + schema["optional_columns"])
        unexpected = set(df.columns) - expected_columns
        if unexpected:
            warnings.append(f"Unexpected columns present: {unexpected}")

        # --------------------------------------------
        # 3. Validate dtypes for columns where types are defined
        # --------------------------------------------
        for col, expected_types in schema["types"].items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type not in expected_types:
                    warnings.append(
                        f"Column '{col}' dtype mismatch → expected any of {expected_types}, got {actual_type}"
                    )

        # --------------------------------------------
        # 4. Hard uniqueness validation
        # --------------------------------------------
        for key_tuple in schema.get("unique_keys", []):
            if all(k in df.columns for k in key_tuple):
                duplicates = df.duplicated(subset=list(key_tuple)).sum()
                if duplicates > 0:
                    errors.append(
                        f"Duplicate values found for unique key {key_tuple}: count={duplicates}"
                    )

        # --------------------------------------------
        # 5. Soft (warning-only) uniqueness validation
        # --------------------------------------------
        for key_tuple in schema.get("soft_unique_keys", []):
            if all(k in df.columns for k in key_tuple):
                duplicates = df.duplicated(subset=list(key_tuple)).sum()
                if duplicates > 0:
                    warnings.append(
                        f"Soft uniqueness check failed for key {key_tuple}: {duplicates} duplicates found "
                        f"(acceptable but flagged)"
                    )

        # --------------------------------------------
        # 6. Completely empty columns
        # --------------------------------------------
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            warnings.append(f"Completely empty columns detected: {empty_cols}")

        # --------------------------------------------
        # 7. Business constraint validation
        # --------------------------------------------
        business_warnings = self._validate_business_constraints(df, schema_name)
        warnings.extend(business_warnings)

        # --------------------------------------------
        # LOGGING AND FINAL DECISION
        # --------------------------------------------
        if errors:
            err_msg = f"\n{schema_name} validation FAILED:\n" + "\n".join(f"  ❌ {e}" for e in errors)
            logger.error(err_msg)
            raise ValueError(err_msg)

        if warnings:
            warn_msg = f"\n{schema_name} validation WARNINGS:\n" + "\n".join(f"  ⚠️  {w}" for w in warnings)
            logger.warning(warn_msg)

        logger.info(f"{schema_name}: Schema validation passed ✔")
        return True

    def _validate_business_constraints(self, df: pd.DataFrame, schema_name: str) -> List[str]:
        """
        Validate business logic constraints on values.
        Returns list of warnings (non-blocking).
        """
        warnings = []
        
        # Customers: MRR business rules
        if schema_name == "customers":
            if "MRR" in df.columns:
                invalid_mrr = df[df["MRR"] <= 0]
                if len(invalid_mrr) > 0:
                    pct = len(invalid_mrr) / len(df) * 100
                    warnings.append(
                        f"Found {len(invalid_mrr)} customers ({pct:.1f}%) with MRR ≤ 0 "
                        f"(pipeline will clean these)"
                    )
        
        # All companies: ALEXA_RANK must be positive
        if schema_name in ["customers", "noncustomers"]:
            if "ALEXA_RANK" in df.columns:
                invalid_alexa = df[(df["ALEXA_RANK"] <= 0) & (df["ALEXA_RANK"].notna())]
                if len(invalid_alexa) > 0:
                    warnings.append(
                        f"Found {len(invalid_alexa)} rows with ALEXA_RANK ≤ 0"
                    )
            
            # EMPLOYEE_RANGE valid categories
            if "EMPLOYEE_RANGE" in df.columns:
                valid_ranges = [
                    '1', '2 to 5', '6 to 10', '11 to 25', '26 to 50',
                    '51 to 200', '201 to 1000', '1001 to 10000', '10,001 or more'
                ]
                non_null = df["EMPLOYEE_RANGE"].dropna()
                invalid_ranges = non_null[~non_null.isin(valid_ranges)]
                if len(invalid_ranges) > 0:
                    pct = len(invalid_ranges) / len(non_null) * 100
                    warnings.append(
                        f"Found {len(invalid_ranges)} ({pct:.1f}%) rows with "
                        f"invalid EMPLOYEE_RANGE values"
                    )
        
        # Usage: Action counts must be non-negative
        if schema_name == "usage_actions":
            action_cols = [col for col in df.columns if col.startswith("ACTIONS_")]
            for col in action_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    negative = df[df[col] < 0]
                    if len(negative) > 0:
                        warnings.append(
                            f"Found {len(negative)} negative values in {col}"
                        )
        
        return warnings


class DataQualityProfiler:
    """
    Produce structured data-quality metrics & logs:
    - missingness
    - duplicate rows
    - memory usage
    - basic numeric summaries

    Intended to provide DS visibility, not strict validation.
    """

    def profile(self, df: pd.DataFrame, name: str) -> Dict:
        profile = {
            "name": name,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "duplicates": {
                "n_duplicate_rows": int(df.duplicated().sum()),
                "pct_duplicate_rows": float(df.duplicated().sum() / len(df) * 100),
            },
            "missing_values": {},
            "column_types": df.dtypes.astype(str).to_dict(),
            "numeric_summary": {},
        }

        # Missingness
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                profile["missing_values"][col] = {
                    "count": int(n_missing),
                    "percentage": float(n_missing / len(df) * 100),
                }

        # Summary stats for numerics
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            profile["numeric_summary"] = df[numeric_cols].describe().to_dict()

        # Log summary
        self._log_profile(profile)
        return profile

    def _log_profile(self, profile: Dict) -> None:
        logger.info("\n" + "=" * 70)
        logger.info(f"DATA QUALITY PROFILE → {profile['name']}")
        logger.info("=" * 70)
        logger.info(f"Rows: {profile['n_rows']:,}")
        logger.info(f"Columns: {profile['n_columns']}")
        logger.info(f"Memory Usage: {profile['memory_usage_mb']:.3f} MB")
        logger.info(
            f"Duplicate Rows: {profile['duplicates']['n_duplicate_rows']} "
            f"({profile['duplicates']['pct_duplicate_rows']:.2f}%)"
        )

        if profile["missing_values"]:
            logger.info("\nMissing Values:")
            for col, info in profile["missing_values"].items():
                logger.info(f"  {col}: {info['count']} ({info['percentage']:.1f}%)")
        else:
            logger.info("Missing Values: None")

        logger.info("=" * 70 + "\n")

class DataDeduplicator:
    """
    Deduplicate rows intelligently based on key columns.

    Strategies:
    - most_complete: keep row with highest non-null count
    - first: keep first occurrence
    - last: keep last occurrence
    """

    def __init__(self, key_column: Union[str, Tuple[str, ...]] = "id"):
        self.key_column = key_column

    def deduplicate(
        self,
        df: pd.DataFrame,
        key_columns: Union[str, List[str]] = 'id',
        strategy: str = 'most_complete',
        save_duplicates: bool = True,
        df_name: str = 'data'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows based on key columns.
        
        Args:
            df: DataFrame to deduplicate
            key_columns: Column(s) to check for duplicates
            strategy: Deduplication strategy
            save_duplicates: Whether to save removed duplicates
            df_name: Name for the saved file
            
        Returns:
            Deduplicated DataFrame
        """
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        
        initial_count = len(df)
        
        # Find duplicates
        duplicates_mask = df.duplicated(subset=key_columns, keep=False)
        duplicates = df[duplicates_mask].copy()
        duplicate_count = duplicates[key_columns[0]].nunique() if len(duplicates) > 0 else 0
        
        if duplicate_count == 0:
            logger.info(f"✓ No duplicates found for key {key_columns}")
            return df
        
        logger.warning(
            f"\n{'='*70}\n"
            f"⚠️  DUPLICATES FOUND: {df_name}\n"
            f"{'='*70}\n"
            f"Found {len(duplicates)} duplicate rows for key={key_columns}\n"
            f"Unique IDs affected: {duplicate_count}\n"
            f"Strategy '{strategy}' will be applied.\n"
        )
        
        # **NEW: Save duplicates before removing**
        if save_duplicates and len(duplicates) > 0:
            output_dir = Path('artifacts/data_quality')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sort by key and add metadata
            duplicates_sorted = duplicates.sort_values(by=key_columns)
            duplicates_sorted.insert(0, '_duplicate_group', 
                                    duplicates_sorted.groupby(key_columns).ngroup())
            
            output_path = output_dir / f'duplicates_{df_name}.csv'
            duplicates_sorted.to_csv(output_path, index=False)
            
            logger.info(f"💾 Saved {len(duplicates)} duplicate rows to: {output_path}")
            
            # Show sample
            sample = duplicates_sorted[key_columns + ['_duplicate_group']].head(10)
            logger.warning(f"\nSample duplicate groups:\n{sample}\n")
        
        # Apply deduplication strategy
        if strategy == 'first':
            df_clean = df.drop_duplicates(subset=key_columns, keep='first')
        elif strategy == 'last':
            df_clean = df.drop_duplicates(subset=key_columns, keep='last')
        elif strategy == 'most_complete':
            df_clean = self._keep_most_complete(df, key_columns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        removed_count = initial_count - len(df_clean)
        
        logger.warning(f"🗑️  Removed {removed_count} duplicate rows")
        logger.info(f"✓ Retained {len(df_clean)} unique rows ({len(df_clean)/initial_count*100:.2f}%)")
        logger.warning(f"{'='*70}")
        
        return df_clean
        
    def _keep_most_complete(self, df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
        """
        Keep the most complete row for each duplicate group.
        
        "Most complete" = row with fewest missing values.
        
        Args:
            df: DataFrame with duplicates
            key_columns: Columns that define duplicates
            
        Returns:
            DataFrame with duplicates removed (keeping most complete)
        """
        def completeness_score(row):
            """Calculate how complete a row is (higher = more complete)."""
            score = 0
            for val in row:
                if pd.notna(val) and val != '' and val != 0:
                    score += 1
            return score
        
        # Add completeness score to each row
        df = df.copy()
        df['_completeness'] = df.apply(completeness_score, axis=1)
        
        # Sort by key columns and completeness (descending)
        df_sorted = df.sort_values(
            by=key_columns + ['_completeness'],
            ascending=[True] * len(key_columns) + [False]
        )
        
        # Keep first (most complete) for each key
        df_clean = df_sorted.drop_duplicates(subset=key_columns, keep='first')
        
        # Remove helper column
        df_clean = df_clean.drop('_completeness', axis=1)
        
        return df_clean
       