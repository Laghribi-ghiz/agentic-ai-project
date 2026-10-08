from graph.state import AgentState


def ml_agent(state: AgentState) -> AgentState:

    problem_type = state["problem_type"]
    recommended_models = state["recommended_models"]
    evaluation_metrics = state["evaluation_metrics"]

    if problem_type == "binary classification":
        selected_models = [
            "LogisticRegression",
            "RandomForestClassifier"
        ]
    else:
        selected_models = recommended_models

    ml_strategy = {
        "problem_type": problem_type,
        "models": selected_models,
        "metrics": evaluation_metrics,
    }

    return {
        **state,
        "ml_strategy": ml_strategy,
    }