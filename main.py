from unittest import result
from tools.model_tools import save_model

from graph.state import AgentState
from agents.data_agent import data_agent
from agents.analyst_agent import analyst_agent
from agents.ml_agent import ml_agent

from langgraph.graph import StateGraph, START, END

from tools.ml_tools import (
    preprocess_telco_dataset,
    split_dataset,
    train_logistic_regression,
    train_random_forest,
    compare_models,
)
from agents.evaluation_agent import evaluation_agent
from tools.mlflow_tools import setup_mlflow, log_model_experiment


def train_and_compare_agent(state: AgentState) -> AgentState:
    dataset = preprocess_telco_dataset(
        state["dataset_path"],
        state["target_column"],
    )

    X = dataset["X"]
    y = dataset["y"]

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    logistic_result = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    rf_result = train_random_forest(
        X_train, X_test, y_train, y_test
    )

    comparison = compare_models(logistic_result, rf_result)

    setup_mlflow()

    log_model_experiment(
        model_name="LogisticRegression",
        metrics=logistic_result["metrics"],
        parameters={"max_iter": 1000},
    )

    log_model_experiment(
        model_name="RandomForest",
        metrics=rf_result["metrics"],
        parameters={
            "n_estimators": 200,
            "random_state": 42,
        },
    )
    if comparison["best_model"] == "LogisticRegression":
        best_model_object = logistic_result["model"]
    else:
     best_model_object = rf_result["model"]

    model_path = save_model(
    best_model_object,
    comparison["best_model"]
)

    print(f"Best model saved to: {model_path}", flush=True)
    

    return {
        **state,
        "model_comparison": comparison,
    }
  

def main():

    workflow = StateGraph(AgentState)

    workflow.add_node("data_agent", data_agent)
    workflow.add_node("analyst_agent", analyst_agent)
    workflow.add_node("ml_agent", ml_agent)
    workflow.add_node("train_and_compare", train_and_compare_agent)
    workflow.add_node("evaluation_agent", evaluation_agent)

    workflow.add_edge(START, "data_agent")
    workflow.add_edge("data_agent", "analyst_agent")
    workflow.add_edge("analyst_agent", "ml_agent")
    workflow.add_edge("ml_agent", "train_and_compare")
    workflow.add_edge("train_and_compare", "evaluation_agent")
    workflow.add_edge("evaluation_agent", END)

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

    
    print("\n=== EVALUATION SUMMARY ===")
    print(result["evaluation_summary"])

if __name__ == "__main__":
    main()