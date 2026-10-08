from unittest import result

from graph.state import AgentState
from agents.data_agent import data_agent
from agents.analyst_agent import analyst_agent
from agents.ml_agent import ml_agent

from langgraph.graph import StateGraph, START, END


def main():

    workflow = StateGraph(AgentState)

    workflow.add_node("data_agent", data_agent)
    workflow.add_node("analyst_agent", analyst_agent)
    workflow.add_node("ml_agent", ml_agent)

    workflow.add_edge(START, "data_agent")
    workflow.add_edge("data_agent", "analyst_agent")
    workflow.add_edge("analyst_agent", "ml_agent")
    workflow.add_edge("ml_agent", END)

    app = workflow.compile()

    initial_state = {
        "dataset_path": "data/raw/Telco-Customer-Churn.csv",
        "target_column": "Churn",

        "dataset_shape": None,
        "columns": None,

        "numerical_columns": None,
        "categorical_columns": None,

        "identifier_columns": None,
        "suspicious_columns": None,

        "missing_values": None,
        "hidden_missing_values": None,
        "target_distribution": None,

        "problem_type": None,

        "data_analysis": None,
        "recommendations": None,
        
        "llm_analysis": None,
        "processed_dataset_path": None,
        "ml_strategy": None,
    }

    result = app.invoke(initial_state)

    print("\n=== DATASET ANALYSIS ===")

    print("Shape:", result["dataset_shape"])

    print("\nNumerical columns:")
    print(result["numerical_columns"])

    print("\nCategorical columns:")
    print(result["categorical_columns"])

    print("\nIdentifier columns:")
    print(result["identifier_columns"])

    print("\nSuspicious columns:")
    print(result["suspicious_columns"])

    print("\nTarget distribution:")
    print(result["target_distribution"])

    print("\nProblem type:")
    print(result["problem_type"])

    print("\nAnalysis:")
    print(result["data_analysis"])

    print("\nRecommendations:")

    for recommendation in result["recommendations"]:
        print("-", recommendation)

    print("\n=== LLM ANALYSIS ===")
    print(result["llm_analysis"])

    print("\n=== ML STRATEGY ===")
    print(result["ml_strategy"])

if __name__ == "__main__":
    main()