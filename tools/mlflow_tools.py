import mlflow


def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco-Customer-Churn")


def log_model_experiment(
    model_name: str,
    metrics: dict,
    parameters: dict | None = None,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)

        if parameters:
            mlflow.log_params(parameters)

        mlflow.log_metrics(metrics)