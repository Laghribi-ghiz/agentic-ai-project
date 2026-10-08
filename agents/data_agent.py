import pandas as pd

from graph.state import AgentState


def detect_identifier_columns(df: pd.DataFrame) -> list:
    """
    Detect columns that are likely identifiers.
    """

    identifiers = []

    for column in df.columns:

        unique_ratio = df[column].nunique() / len(df)

        if unique_ratio > 0.95:
            identifiers.append(column)

    return identifiers


def detect_suspicious_numeric_columns(df: pd.DataFrame) -> list:
    """
    Detect columns stored as strings but potentially representing numbers.
    """

    suspicious = []

    for column in df.select_dtypes(include=["object"]).columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio > 0.90:
            suspicious.append(column)

    return suspicious

def detect_hidden_missing_values(df: pd.DataFrame) -> dict:
    hidden_missing = {}

    for column in df.select_dtypes(include=["object"]).columns:
        count = df[column].astype(str).str.strip().eq("").sum()

        if count > 0:
            hidden_missing[column] = int(count)

    return hidden_missing

def data_agent(state: AgentState) -> AgentState:

    dataset_path = state["dataset_path"]
    target_column = state["target_column"]

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = pd.read_csv(dataset_path)

    # --------------------------------------------------
    # 2. Basic information
    # --------------------------------------------------

    dataset_shape = df.shape
    columns = df.columns.tolist()

    # --------------------------------------------------
    # 3. Identify numerical / categorical columns
    # --------------------------------------------------

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)

    # --------------------------------------------------
    # 4. Detect identifiers
    # --------------------------------------------------

    identifier_columns = detect_identifier_columns(df)

    # Remove identifiers from useful ML features
    categorical_columns = [
        col for col in categorical_columns
        if col not in identifier_columns
    ]

    # --------------------------------------------------
    # 5. Detect suspicious columns
    # --------------------------------------------------

    suspicious_columns = detect_suspicious_numeric_columns(df)
    hidden_missing_values = detect_hidden_missing_values(df)

    # --------------------------------------------------
    # 6. Missing values
    # --------------------------------------------------

    missing_values = df.isnull().sum().to_dict()

    # --------------------------------------------------
    # 7. Target analysis
    # --------------------------------------------------

    target_distribution = {}

    if target_column and target_column in df.columns:

        target_distribution = (
            df[target_column]
            .value_counts()
            .to_dict()
        )

        number_of_classes = df[target_column].nunique()

        if number_of_classes == 2:
            problem_type = "binary classification"

        elif number_of_classes > 2:
            problem_type = "multiclass classification"

        else:
            problem_type = "unknown"

    else:
        problem_type = "unknown"

    # --------------------------------------------------
    # 8. Generate recommendations
    # --------------------------------------------------

    recommendations = []
    if hidden_missing_values:
        recommendations.append(
        f"Hidden missing values detected: {hidden_missing_values}"
    )

    if identifier_columns:
        recommendations.append(
            f"Remove identifier columns from ML features: "
            f"{identifier_columns}"
        )

    if suspicious_columns:
        recommendations.append(
            f"Investigate columns stored as text but "
            f"potentially numeric: {suspicious_columns}"
        )

    if any(value > 0 for value in missing_values.values()):
        recommendations.append(
            "Missing values detected. "
            "Preprocessing is required."
        )

    if problem_type == "binary classification":
        recommendations.append(
            "Use binary classification models."
        )

    # --------------------------------------------------
    # 9. Analysis summary
    # --------------------------------------------------

    data_analysis = (
        f"Dataset contains {dataset_shape[0]} rows "
        f"and {dataset_shape[1]} columns. "
        f"The detected ML problem is {problem_type}. "
        f"{len(numerical_columns)} numerical columns "
        f"and {len(categorical_columns)} categorical columns "
        f"were detected."
    )

    return {
        **state,

        "dataset_shape": dataset_shape,
        "columns": columns,

        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,

        "identifier_columns": identifier_columns,
        "suspicious_columns": suspicious_columns,

        "hidden_missing_values": hidden_missing_values,

        
        "missing_values": missing_values,
        "target_distribution": target_distribution,

        "problem_type": problem_type,

        "data_analysis": data_analysis,
        "recommendations": recommendations,
    }