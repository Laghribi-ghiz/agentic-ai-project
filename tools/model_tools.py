import joblib
from pathlib import Path


def save_model(model, model_name: str) -> str:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_name}.joblib"

    joblib.dump(model, model_path)

    return str(model_path)