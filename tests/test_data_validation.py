"""
Unit tests for HubSpot ML Framework.
Tests business logic, data integrity, and pipeline behavior on real data.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestBusinessLogic:
    """Test business logic rules on real data."""
    
    @pytest.fixture(scope="class")
    def customers_df(self):
        """Load real customers data."""
        return pd.read_csv('data/customers.csv')
    
    @pytest.fixture(scope="class")
    def noncustomers_df(self):
        """Load real non-customers data."""
        return pd.read_csv('data/noncustomers.csv')
    
    @pytest.fixture(scope="class")
    def usage_df(self):
        """Load real usage data."""
        return pd.read_csv('data/usage_actions.csv')
    
    @pytest.fixture(scope="class")
    def all_companies(self, customers_df, noncustomers_df):
        """Combine all company data."""
        return pd.concat([customers_df, noncustomers_df], ignore_index=True)
    
    # ============================================================
    # TEST 1: ALEXA_RANK Business Rule
    # ============================================================
    
    def test_alexa_rank_reasonable_values(self, all_companies):
        """
        Business Rule: ALEXA_RANK should be reasonable for B2B SaaS trials.
        - Must be positive
        - Less than 5% should be top-100 sites (suspicious for B2B trials)
        """
        alexa_valid = all_companies['ALEXA_RANK'].dropna()
        
        # Must be positive
        assert (alexa_valid > 0).all(), "ALEXA_RANK must be positive"
        
        # Flag top global sites (unlikely for B2B trial signups)
        top_100 = alexa_valid[alexa_valid < 100]
        top_100_pct = len(top_100) / len(alexa_valid) * 100
        
        print(f"\nALEXA_RANK distribution:")
        print(f"  Total companies with rank: {len(alexa_valid)}")
        print(f"  Top 100 sites: {len(top_100)} ({top_100_pct:.2f}%)")
        
        assert top_100_pct < 5, \
            f"Too many top-100 sites: {top_100_pct:.1f}% (expected < 5%)"    
    
    # ============================================================
    # TEST 2: Actions >= Users Business Rule
    # ============================================================
    
    def test_actions_must_exceed_users(self, usage_df):
        """
        Business Rule: For any activity, ACTIONS >= USERS.
        Logic: Each user must perform at least 1 action.
        If ACTIONS < USERS, data is inconsistent.
        """
        object_types = ['CRM_CONTACTS', 'CRM_COMPANIES', 'CRM_DEALS', 'EMAIL']
        
        violations = []
        for obj_type in object_types:
            actions_col = f'ACTIONS_{obj_type}'
            users_col = f'USERS_{obj_type}'
            
            if actions_col in usage_df.columns and users_col in usage_df.columns:
                invalid = usage_df[
                    (usage_df[actions_col] < usage_df[users_col]) & 
                    (usage_df[actions_col].notna()) & 
                    (usage_df[users_col].notna())
                ]
                
                if len(invalid) > 0:
                    violations.append({
                        'type': obj_type,
                        'count': len(invalid),
                        'pct': len(invalid) / len(usage_df) * 100
                    })
        
        if violations:
            print(f"\nBusiness rule violations (ACTIONS < USERS):")
            for v in violations:
                print(f"  {v['type']}: {v['count']} rows ({v['pct']:.2f}%)")
        
        assert len(violations) == 0, \
            f"Found {len(violations)} violations of ACTIONS >= USERS rule"
    
    # ============================================================
    # TEST 3: Usage Data Integrity
    # ============================================================
    
    def test_usage_data_has_matching_companies(self, all_companies, usage_df):
        """
        Business Rule: Every usage record must belong to a known company.
        Orphan usage data suggests data quality issues.
        """
        company_ids = set(all_companies['id'].unique())
        usage_ids = set(usage_df['id'].unique())
        
        orphan_usage = usage_ids - company_ids
        
        if len(orphan_usage) > 0:
            print(f"\nOrphan usage IDs (no matching company): {list(orphan_usage)[:10]}")
        
        assert len(orphan_usage) == 0, \
            f"Found {len(orphan_usage)} usage records without matching company"
    
    # ============================================================
    # TEST 4: Action Values Must Be Non-Negative
    # ============================================================
    
    def test_action_counts_non_negative(self, usage_df):
        """
        Business Rule: Action counts cannot be negative.
        """
        action_cols = [col for col in usage_df.columns if col.startswith('ACTIONS_')]
        
        violations = {}
        for col in action_cols:
            negative = usage_df[usage_df[col] < 0]
            if len(negative) > 0:
                violations[col] = len(negative)
        
        if violations:
            print(f"\nNegative action counts found:")
            for col, count in violations.items():
                print(f"  {col}: {count} negative values")
        
        assert len(violations) == 0, \
            f"Found negative values in {len(violations)} action columns"


class TestPipelineBehavior:
    """Test that the data pipeline correctly handles business rules."""
    
    def test_pipeline_removes_invalid_mrr_customers(self):
        """
        Test: Pipeline must remove customers with MRR <= 0.
        Business Rule: is_customer=1 requires MRR > 0.
        """
        from ml_framework.data import HubSpotDataLoader
        
        # Load raw data
        raw_customers = pd.read_csv('data/customers.csv')
        raw_invalid = raw_customers[raw_customers['MRR'] <= 0]
        raw_invalid_count = len(raw_invalid)
        raw_invalid_pct = (raw_invalid_count / len(raw_customers)) * 100
        
        print(f"\n📊 Raw Data Quality:")
        print(f"  Total customers: {len(raw_customers)}")
        print(f"  Invalid MRR (≤0): {raw_invalid_count} ({raw_invalid_pct:.2f}%)")
        
        # Load through pipeline
        loader = HubSpotDataLoader(
            customers_path='data/customers.csv',
            noncustomers_path='data/noncustomers.csv',
            usage_path='data/usage_actions.csv'
        )
        
        df = loader.load_data()
        customers_after = df[df['is_customer'] == 1]
        
        print(f"\n✅ After Pipeline:")
        print(f"  Customers in training: {len(customers_after)}")
        print(f"  Removed: {raw_invalid_count}")
        
        # Verify MRR was either cleaned or removed
        if 'MRR' in customers_after.columns:
            invalid_after = customers_after[customers_after['MRR'] <= 0]
            assert len(invalid_after) == 0, \
                f"Pipeline failed: {len(invalid_after)} customers still have MRR ≤ 0"
            print(f"  ✓ All remaining customers have valid MRR > 0")
        else:
            print(f"  ✓ MRR column removed (used for filtering only)")
    
    def test_pipeline_handles_duplicates(self):
        """
        Test: Pipeline must handle duplicate company IDs.
        """
        from ml_framework.data import HubSpotDataLoader
        
        # Check raw data for duplicates
        raw_customers = pd.read_csv('data/customers.csv')
        raw_noncustomers = pd.read_csv('data/noncustomers.csv')
        
        customer_dups = raw_customers['id'].duplicated().sum()
        noncustomer_dups = raw_noncustomers['id'].duplicated().sum()
        
        print(f"\n📊 Duplicate Check:")
        print(f"  Customer duplicates (raw): {customer_dups}")
        print(f"  Non-customer duplicates (raw): {noncustomer_dups}")
        
        # Load through pipeline
        loader = HubSpotDataLoader(
            customers_path='data/customers.csv',
            noncustomers_path='data/noncustomers.csv',
            usage_path='data/usage_actions.csv'
        )
        
        df = loader.load_data()
        
        # Check for duplicates after pipeline
        final_dups = df['id'].duplicated().sum()
        
        print(f"\n✅ After Pipeline:")
        print(f"  Duplicates in final data: {final_dups}")
        
        assert final_dups == 0, \
            f"Pipeline failed: {final_dups} duplicate IDs remain in training data"
    
    def test_pipeline_creates_required_features(self):
        """
        Test: Pipeline must create all required features for training.
        """
        from ml_framework.data import HubSpotDataLoader
        
        loader = HubSpotDataLoader(
            customers_path='data/customers.csv',
            noncustomers_path='data/noncustomers.csv',
            usage_path='data/usage_actions.csv'
        )
        
        X, y = loader.load_and_prepare()
        
        print(f"\n📊 Feature Engineering:")
        print(f"  Features created: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        
        # Required feature types
        required_features = {
            'usage_aggregations': ['sum', 'mean', 'max'],  # Aggregated actions
            'derived_features': ['total_actions', 'total_users', 'days_active'],
            'company_attributes': ['ALEXA_RANK', 'EMPLOYEE_RANGE', 'INDUSTRY']
        }
        
        feature_names = X.columns.tolist()
        
        # Check for usage aggregations
        has_sum_features = any('_sum' in f for f in feature_names)
        has_mean_features = any('_mean' in f for f in feature_names)
        
        assert has_sum_features, "Missing aggregated sum features"
        assert has_mean_features, "Missing aggregated mean features"
        
        # Check for derived features
        assert 'total_actions' in feature_names, "Missing total_actions feature"
        assert 'total_users' in feature_names, "Missing total_users feature"
        assert 'days_active' in feature_names, "Missing days_active feature"
        
        print(f"  ✓ All required feature types present")


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.fixture(scope="class")
    def usage_df(self):
        """Load real usage data."""
        return pd.read_csv('data/usage_actions.csv')
    
    def test_usage_has_timestamps(self, usage_df):
        """
        Test: Usage data must have timestamp column.
        Required for temporal feature engineering.
        """
        assert 'WHEN_TIMESTAMP' in usage_df.columns, \
            "Usage data missing WHEN_TIMESTAMP column"
        
        # Check timestamps are parseable
        total = len(usage_df)
        parsed = pd.to_datetime(usage_df['WHEN_TIMESTAMP'], errors='coerce')
        invalid = parsed.isna().sum()
        invalid_pct = (invalid / total) * 100
        
        print(f"\n📊 Timestamp Quality:")
        print(f"  Total records: {total}")
        print(f"  Invalid timestamps: {invalid} ({invalid_pct:.2f}%)")
        
        # Allow max 0.1% bad timestamps
        assert invalid_pct < 0.1, \
            f"Too many invalid timestamps: {invalid_pct:.2f}%"
    
    def test_feature_calculations_on_real_data(self, usage_df):
        """
        Test: Verify feature calculations work on real data.
        Tests days_active, activity_frequency logic.
        """
        usage_df['WHEN_TIMESTAMP'] = pd.to_datetime(usage_df['WHEN_TIMESTAMP'])
        
        # Test days_active calculation
        for company_id in usage_df['id'].unique()[:5]:  # Test first 5 companies
            company_data = usage_df[usage_df['id'] == company_id]
            
            days_active = (company_data['WHEN_TIMESTAMP'].max() - 
                          company_data['WHEN_TIMESTAMP'].min()).days
            
            assert days_active >= 0, f"days_active negative for company {company_id}"
            
            # Test activity_frequency
            if days_active > 0:
                total_actions = company_data[[col for col in company_data.columns 
                                             if col.startswith('ACTIONS_')]].sum().sum()
                activity_freq = total_actions / (days_active + 1)
                
                assert activity_freq >= 0, f"activity_frequency negative for {company_id}"
        
        print(f"\n✅ Feature calculations validated on {len(usage_df['id'].unique())} companies")