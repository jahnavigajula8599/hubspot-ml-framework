#!/usr/bin/env python3
"""
Entity Relationship Diagram Generator for HubSpot ML Framework

This script generates visual ERD diagrams showing:
1. Data Model (Company, UsageAction, AggregatedFeatures)
2. ML Pipeline Components (Trainer, Model, Predictor, etc.)
3. Complete System Architecture

Dependencies:
    - graphviz (pip install graphviz)
"""

import os
from pathlib import Path


def create_data_model_erd():
    """Create ERD for the data model (Company, UsageAction, AggregatedFeatures)"""
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz not installed. Run: pip install graphviz")
        return None

    dot = Digraph(comment='HubSpot ML Framework - Data Model ERD')
    dot.attr(rankdir='LR', size='12,8')
    dot.attr('node', shape='record', style='filled', fillcolor='lightblue')

    # Company entity
    company_attrs = (
        '<f0> COMPANY|'
        '<f1> id: int (PK)|'
        '<f2> ALEXA_RANK: float|'
        '<f3> EMPLOYEE_RANGE: string|'
        '<f4> INDUSTRY: string|'
        '<f5> is_customer: int (0/1)|'
        '<f6> CLOSEDATE: datetime*|'
        '<f7> MRR: float*'
    )
    dot.node('company', company_attrs, fillcolor='#90EE90')

    # UsageAction entity
    usage_attrs = (
        '<f0> USAGE_ACTION|'
        '<f1> id: int (FK)|'
        '<f2> WHEN_TIMESTAMP: datetime|'
        '<f3> ACTIONS_CRM_CONTACTS: int|'
        '<f4> ACTIONS_CRM_COMPANIES: int|'
        '<f5> ACTIONS_CRM_DEALS: int|'
        '<f6> ACTIONS_EMAIL: int|'
        '<f7> USERS_CRM_CONTACTS: int|'
        '<f8> USERS_CRM_COMPANIES: int|'
        '<f9> USERS_CRM_DEALS: int|'
        '<f10> USERS_EMAIL: int'
    )
    dot.node('usage', usage_attrs, fillcolor='#FFB6C1')

    # AggregatedFeatures entity
    agg_attrs = (
        '<f0> AGGREGATED_FEATURES|'
        '<f1> id: int (FK)|'
        '<f2> total_actions: float|'
        '<f3> total_users: float|'
        '<f4> actions_per_user: float|'
        '<f5> days_active: int|'
        '<f6> activity_frequency: float|'
        '<f7> ACTIONS_CRM_*_sum: float|'
        '<f8> ACTIONS_CRM_*_mean: float|'
        '<f9> ACTIONS_CRM_*_max: float|'
        '<f10> ACTIONS_CRM_*_std: float|'
        '<f11> USERS_CRM_*_sum: float|'
        '<f12> USERS_CRM_*_mean: float|'
        '<f13> USERS_CRM_*_max: float'
    )
    dot.node('aggregated', agg_attrs, fillcolor='#FFDAB9')

    # Relationships
    dot.edge('company:f1', 'usage:f1', label='1:N\ngenerates', color='blue', fontcolor='blue')
    dot.edge('company:f1', 'aggregated:f1', label='1:1\nhas', color='green', fontcolor='green')

    # Add legend
    legend = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        '<TR><TD COLSPAN="2" BGCOLOR="gray"><B>Legend</B></TD></TR>'
        '<TR><TD>*</TD><TD ALIGN="left">Customers only</TD></TR>'
        '<TR><TD>PK</TD><TD ALIGN="left">Primary Key</TD></TR>'
        '<TR><TD>FK</TD><TD ALIGN="left">Foreign Key</TD></TR>'
        '<TR><TD BGCOLOR="#90EE90">Green</TD><TD ALIGN="left">Core Entity</TD></TR>'
        '<TR><TD BGCOLOR="#FFB6C1">Pink</TD><TD ALIGN="left">Activity Entity</TD></TR>'
        '<TR><TD BGCOLOR="#FFDAB9">Peach</TD><TD ALIGN="left">Derived Entity</TD></TR>'
        '</TABLE>>'
    )
    dot.node('legend', legend, shape='plaintext')

    return dot


def create_ml_pipeline_erd():
    """Create ERD for ML pipeline components"""
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz not installed. Run: pip install graphviz")
        return None

    dot = Digraph(comment='HubSpot ML Framework - ML Pipeline ERD')
    dot.attr(rankdir='TB', size='14,10')
    dot.attr('node', shape='box', style='filled,rounded')

    # Config
    dot.node('config', 'CONFIG\n\nexperiment_name\ndata_paths\nmodel_type\nhyperparameters\nrandom_seed',
             fillcolor='#FFE4B5')

    # Trainer
    dot.node('trainer', 'TRAINER\n\nconfig\ndata_loader\nmodel\nfeature_engineer\nartifact_dir',
             fillcolor='#87CEEB')

    # Data Loading Components
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Data Loading Components', style='filled', color='lightgray')
        c.node('data_loader', 'DATA_LOADER\n\ncustomers_path\nnoncustomers_path\nusage_path',
               fillcolor='#98FB98')
        c.node('validator', 'SCHEMA_VALIDATOR\n\nrequired_columns\ncolumn_types',
               fillcolor='#98FB98')
        c.node('profiler', 'DATA_QUALITY_PROFILER\n\nmissing_values\nduplicates',
               fillcolor='#98FB98')
        c.node('dedup', 'DATA_DEDUPLICATOR\n\nstrategy',
               fillcolor='#98FB98')

    # Feature Engineering Components
    with dot.subgraph(name='cluster_features') as c:
        c.attr(label='Feature Engineering Components', style='filled', color='lightgray')
        c.node('feature_eng', 'FEATURE_ENGINEER\n\nscaling_method\nnumeric_features\ncategorical_features',
               fillcolor='#DDA0DD')
        c.node('employee_enc', 'EMPLOYEE_ENCODER\n\nordinal_mapping (1-9)',
               fillcolor='#DDA0DD')
        c.node('industry_enc', 'INDUSTRY_ENCODER\n\nmin_frequency\nindustry_mapping',
               fillcolor='#DDA0DD')
        c.node('missing_handler', 'MISSING_VALUE_HANDLER\n\nstrategy\nimpute_values',
               fillcolor='#DDA0DD')

    # Model Components
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Model Components', style='filled', color='lightgray')
        c.node('model', 'BASE_MODEL\n\n_is_fitted\nmodel_type',
               fillcolor='#F0E68C')
        c.node('lr', 'LogisticRegression', fillcolor='#FFFACD')
        c.node('rf', 'RandomForest', fillcolor='#FFFACD')
        c.node('xgb', 'XGBoost', fillcolor='#FFFACD')
        c.node('lgbm', 'LightGBM', fillcolor='#FFFACD')

    # Evaluator
    dot.node('evaluator', 'EVALUATOR\n\nmodel\ny_true\ny_pred\ny_proba',
             fillcolor='#87CEEB')

    # Predictor
    dot.node('predictor', 'PREDICTOR\n\nmodel\nfeature_engineer\npreprocessors',
             fillcolor='#FFA07A')

    # API
    dot.node('api', 'FAST_API\n\nendpoints\nmodel_service',
             fillcolor='#FFA07A')

    # Artifact
    dot.node('artifact', 'TRAINED_ARTIFACT\n\nmodel.joblib\npreprocessors.joblib\nconfig.yaml\nmetrics.json',
             fillcolor='#D3D3D3')

    # Relationships
    # Config relationships
    dot.edge('config', 'trainer', label='configures')

    # Trainer relationships
    dot.edge('trainer', 'data_loader', label='uses')
    dot.edge('trainer', 'feature_eng', label='uses')
    dot.edge('trainer', 'model', label='trains')
    dot.edge('trainer', 'evaluator', label='uses')

    # Data loader relationships
    dot.edge('data_loader', 'validator', label='validates with')
    dot.edge('data_loader', 'profiler', label='profiles with')
    dot.edge('data_loader', 'dedup', label='deduplicates with')

    # Feature engineer relationships
    dot.edge('feature_eng', 'employee_enc', label='uses')
    dot.edge('feature_eng', 'industry_enc', label='uses')
    dot.edge('feature_eng', 'missing_handler', label='uses')

    # Model inheritance
    dot.edge('model', 'lr', label='<<implements>>', style='dashed')
    dot.edge('model', 'rf', label='<<implements>>', style='dashed')
    dot.edge('model', 'xgb', label='<<implements>>', style='dashed')
    dot.edge('model', 'lgbm', label='<<implements>>', style='dashed')

    # Predictor relationships
    dot.edge('predictor', 'model', label='uses')
    dot.edge('predictor', 'feature_eng', label='uses')
    dot.edge('predictor', 'employee_enc', label='uses')
    dot.edge('predictor', 'industry_enc', label='uses')
    dot.edge('predictor', 'missing_handler', label='uses')

    # API relationships
    dot.edge('api', 'predictor', label='uses')

    # Artifact relationships
    dot.edge('trainer', 'artifact', label='saves', color='green')
    dot.edge('artifact', 'predictor', label='loads', color='blue')

    return dot


def create_system_architecture_diagram():
    """Create complete system architecture diagram"""
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz not installed. Run: pip install graphviz")
        return None

    dot = Digraph(comment='HubSpot ML Framework - System Architecture')
    dot.attr(rankdir='TB', size='16,12')
    dot.attr('node', shape='box', style='filled,rounded')

    # Data Sources
    with dot.subgraph(name='cluster_sources') as c:
        c.attr(label='Data Sources', style='filled', color='#E0F7FA')
        c.node('customers_csv', 'customers.csv\n(200 records)', fillcolor='#80DEEA')
        c.node('noncustomers_csv', 'noncustomers.csv\n(5,003 records)', fillcolor='#80DEEA')
        c.node('usage_csv', 'usage_actions.csv\n(25,387 records)', fillcolor='#80DEEA')

    # Data Loading & Validation
    with dot.subgraph(name='cluster_loading') as c:
        c.attr(label='Data Loading & Validation', style='filled', color='#F1F8E9')
        c.node('loader', 'HubSpotDataLoader', fillcolor='#C5E1A5')
        c.node('validation', 'Schema Validation\nQuality Profiling\nDeduplication', fillcolor='#C5E1A5')

    # Feature Engineering
    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='Preprocessing Pipeline', style='filled', color='#F3E5F5')
        c.node('preprocessing', 'Preprocessing Pipeline\n\n1. Employee Range Encoding\n2. Industry Encoding\n3. Missing Value Handling\n4. Feature Scaling',
               fillcolor='#CE93D8')

    # Model Training
    with dot.subgraph(name='cluster_training') as c:
        c.attr(label='Model Training', style='filled', color='#FFF3E0')
        c.node('training', 'Model Training\n\nLogisticRegression\nRandomForest\nXGBoost\nLightGBM',
               fillcolor='#FFB74D')
        c.node('evaluation', 'Model Evaluation\n\nMetrics\nVisualization\nFeature Importance',
               fillcolor='#FFB74D')

    # Artifacts
    with dot.subgraph(name='cluster_artifacts') as c:
        c.attr(label='Trained Artifacts', style='filled', color='#ECEFF1')
        c.node('artifacts', 'Artifact Directory\n\nmodel.joblib\npreprocessors.joblib\nconfig.yaml\nmetrics.json\nplots/',
               fillcolor='#B0BEC5')

    # Serving
    with dot.subgraph(name='cluster_serving') as c:
        c.attr(label='Model Serving', style='filled', color='#E8F5E9')
        c.node('predictor', 'Predictor', fillcolor='#81C784')
        c.node('api', 'FastAPI\n\nPOST /predict\nPOST /predict/single\nGET /health',
               fillcolor='#81C784')

    # MLflow
    dot.node('mlflow', 'MLflow Tracking\n\nExperiments\nRuns\nMetrics\nArtifacts', fillcolor='#FFCCBC')

    # Users
    dot.node('users', 'API Clients\n\nHTTP Requests', shape='ellipse', fillcolor='#CFD8DC')

    # Data flow
    dot.edge('customers_csv', 'loader')
    dot.edge('noncustomers_csv', 'loader')
    dot.edge('usage_csv', 'loader')

    dot.edge('loader', 'validation')
    dot.edge('validation', 'preprocessing')
    dot.edge('preprocessing', 'training')
    dot.edge('training', 'evaluation')
    dot.edge('evaluation', 'artifacts')
    dot.edge('training', 'mlflow', label='log metrics', style='dashed')
    dot.edge('artifacts', 'mlflow', label='save artifacts', style='dashed')

    dot.edge('artifacts', 'predictor', label='load', color='blue')
    dot.edge('predictor', 'api')
    dot.edge('users', 'api', label='HTTP POST', color='green')
    dot.edge('api', 'users', label='JSON Response', color='green')

    return dot


def main():
    """Generate all ERD diagrams"""
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'docs' / 'diagrams'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Entity Relationship Diagrams...")
    print(f"Output directory: {output_dir}")

    # Generate Data Model ERD
    print("\n1. Generating Data Model ERD...")
    data_erd = create_data_model_erd()
    if data_erd:
        output_path = output_dir / 'data_model_erd'
        data_erd.render(output_path, format='png', cleanup=True)
        print(f"   ✓ Saved to: {output_path}.png")

        # Also save as SVG for scalability
        data_erd.render(output_dir / 'data_model_erd', format='svg', cleanup=True)
        print(f"   ✓ Saved to: {output_path}.svg")

    # Generate ML Pipeline ERD
    print("\n2. Generating ML Pipeline ERD...")
    ml_erd = create_ml_pipeline_erd()
    if ml_erd:
        output_path = output_dir / 'ml_pipeline_erd'
        ml_erd.render(output_path, format='png', cleanup=True)
        print(f"   ✓ Saved to: {output_path}.png")

        ml_erd.render(output_dir / 'ml_pipeline_erd', format='svg', cleanup=True)
        print(f"   ✓ Saved to: {output_path}.svg")

    # Generate System Architecture Diagram
    print("\n3. Generating System Architecture Diagram...")
    arch_diagram = create_system_architecture_diagram()
    if arch_diagram:
        output_path = output_dir / 'system_architecture'
        arch_diagram.render(output_path, format='png', cleanup=True)
        print(f"   ✓ Saved to: {output_path}.png")

        arch_diagram.render(output_dir / 'system_architecture', format='svg', cleanup=True)
        print(f"   ✓ Saved to: {output_path}.svg")

    print("\n" + "="*60)
    print("✓ All diagrams generated successfully!")
    print("="*60)
    print(f"\nView the diagrams in: {output_dir}")
    print("\nGenerated files:")
    print("  - data_model_erd.png/svg")
    print("  - ml_pipeline_erd.png/svg")
    print("  - system_architecture.png/svg")


if __name__ == '__main__':
    main()
