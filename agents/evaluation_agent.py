from graph.state import AgentState


def evaluation_agent(state: AgentState) -> AgentState:
    comparison = state["model_comparison"]

    best_model = comparison["best_model"]
    best_score = comparison["best_score"]
    selection_metric = comparison["selection_metric"]

    evaluation_summary = (
        f"Best model: {best_model}. "
        f"Selection metric: {selection_metric}. "
        f"Score: {best_score:.4f}."
    )

    return {
        **state,
        "best_model": best_model,
        "evaluation_summary": evaluation_summary,
    }