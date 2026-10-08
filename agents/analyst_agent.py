from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class AnalystOutput(BaseModel):
    data_quality_issues: list[str] = Field(
        description="Only data quality problems supported by the provided dataset profiling."
    )

    features_to_exclude: list[str] = Field(
        description="Features that should be excluded from ML training."
    )

    features_to_transform: list[str] = Field(
        description="Features that should be transformed or converted before training."
    )

    preprocessing_steps: list[str] = Field(
        description="Concrete preprocessing steps supported by the dataset analysis."
    )

    recommended_models: list[str] = Field(
        description="Recommended machine learning models for the detected problem type."
    )

    evaluation_metrics: list[str] = Field(
        description="Recommended evaluation metrics."
    )

    analysis_confidence: str = Field(
        description="High, Medium, or Low confidence in the analysis."
    )


def analyst_agent(state):

    prompt = f"""
You are an expert Machine Learning Data Analyst.

Analyze ONLY the information provided below.

Dataset shape:
{state["dataset_shape"]}

Numerical columns:
{state["numerical_columns"]}

Categorical columns:
{state["categorical_columns"]}

Identifier columns:
{state["identifier_columns"]}

Suspicious columns:
{state["suspicious_columns"]}

Missing values detected by standard null check:
{state["missing_values"]}

Hidden missing values detected by whitespace/empty-string check:
{state["hidden_missing_values"]}

Target distribution:
{state["target_distribution"]}

Problem type:
{state["problem_type"]}

Existing recommendations:
{state["recommendations"]}


IMPORTANT RULES:

1. Do NOT invent statistics or data quality problems.

2. Distinguish between standard missing values and hidden missing values.
   If standard missing_values are zero but hidden_missing_values contains
   values, report the hidden missing values as a real data quality issue.
3. Never say that there are no missing values if hidden_missing_values
   contains any values.

4. Identifier columns should normally be excluded from ML features.



5. A suspicious numeric column should normally be TRANSFORMED or
   CONVERTED, not automatically excluded.

6. Do NOT exclude TotalCharges simply because it is stored as text.
   It should be converted to numeric if appropriate.

7. Do not recommend deleting rows unless the provided information
   clearly justifies it.

8. Separate features to EXCLUDE from features to TRANSFORM.

9. Recommendations must be consistent with the information provided.

10. The problem type is {state["problem_type"]}.
   Recommend models appropriate for this problem.

Return a structured analysis.
"""

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    structured_llm = llm.with_structured_output(AnalystOutput)

    response = structured_llm.invoke(prompt)

    return {
        **state,
        "llm_analysis": response.model_dump_json(indent=2),

        "data_quality_issues": response.data_quality_issues,
        "features_to_exclude": response.features_to_exclude,
        "features_to_transform": response.features_to_transform,
        "preprocessing_steps": response.preprocessing_steps,
        "recommended_models": response.recommended_models,
        "evaluation_metrics": response.evaluation_metrics,
        "analysis_confidence": response.analysis_confidence,
    }