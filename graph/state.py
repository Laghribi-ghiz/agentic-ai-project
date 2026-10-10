from typing import TypedDict, Optional


class AgentState(TypedDict):
    # =========================
    # Dataset input
    # =========================
    dataset_path: str
    target_column: Optional[str]

    # =========================
    # Dataset profiling
    # =========================
    dataset_shape: Optional[tuple]
    columns: Optional[list]

    numerical_columns: Optional[list]
    categorical_columns: Optional[list]

    identifier_columns: Optional[list]
    suspicious_columns: Optional[list]

    missing_values: Optional[dict]
    hidden_missing_values: Optional[dict]
    target_distribution: Optional[dict]

    problem_type: Optional[str]

    # =========================
    # Deterministic Data Agent
    # =========================
    data_analysis: Optional[str]
    recommendations: Optional[list]

    # =========================
    processed_dataset_path: Optional[str]
    # LLM Analyst Agent
    # =========================
    llm_analysis: Optional[str]

    data_quality_issues: Optional[list]
    features_to_exclude: Optional[list]
    features_to_transform: Optional[list]
    preprocessing_steps: Optional[list]

    recommended_models: Optional[list]
    evaluation_metrics: Optional[list]

    analysis_confidence: Optional[str]
    ml_strategy: Optional[dict]

    model_comparison: Optional[dict]
    best_model: Optional[str]
    evaluation_summary: Optional[str]