import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier


def preprocess_telco_dataset(
    dataset_path: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Remove hidden missing values
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Remove identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Remove rows with missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return {
    "X": X,
    "y": y,
    "rows_before": len(pd.read_csv(dataset_path)),
    "rows_after": len(X),
    "removed_rows": len(pd.read_csv(dataset_path)) - len(X),
    "features": X.columns.tolist(),
}
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            pos_label="Yes"
        ),
        "recall": recall_score(
            y_test,
            predictions,
            pos_label="Yes"
        ),
        "f1": f1_score(
            y_test,
            predictions,
            pos_label="Yes"
        ),
        "roc_auc": roc_auc_score(
            y_test.map({"No": 0, "Yes": 1}),
            probabilities
        )
    }

    return {
        "model": model,
        "metrics": metrics
    }
def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            pos_label="Yes"
        ),
        "recall": recall_score(
            y_test,
            predictions,
            pos_label="Yes"
        ),
        "f1": f1_score(
            y_test,
            predictions,
            pos_label="Yes"
        ),
        "roc_auc": roc_auc_score(
            y_test.map({"No": 0, "Yes": 1}),
            probabilities
        )
    }

    return {
        "model": model,
        "metrics": metrics
    }
def compare_models(logistic_result: dict, rf_result: dict) -> dict:
    results = {
        "LogisticRegression": logistic_result["metrics"],
        "RandomForest": rf_result["metrics"],
    }

    best_model = max(
        results,
        key=lambda name: results[name]["roc_auc"]
    )

    return {
        "results": results,
        "best_model": best_model,
        "selection_metric": "roc_auc",
        "best_score": results[best_model]["roc_auc"],
    }


if __name__ == "__main__":

    result = preprocess_telco_dataset(
        "data/raw/Telco-Customer-Churn.csv",
        "Churn"
    )

    X = result["X"]
    y = result["y"]

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("X train:", X_train.shape)
    print("X test:", X_test.shape)
    print("y train:", y_train.shape)
    print("y test:", y_test.shape)

    print("\nTarget distribution - train:")
    print(y_train.value_counts())

    print("\nTarget distribution - test:")
    print(y_test.value_counts())
    result = train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n=== LOGISTIC REGRESSION ===")

for metric, value in result["metrics"].items():
    print(f"{metric}: {value:.4f}")

    rf_result = train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n=== RANDOM FOREST ===")

for metric, value in rf_result["metrics"].items():
    print(f"{metric}: {value:.4f}")

    comparison = compare_models(result, rf_result)

print("\n=== MODEL COMPARISON ===")

for model_name, metrics in comparison["results"].items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

print("\n=== BEST MODEL ===")
print("Model:", comparison["best_model"])
print("Selection metric:", comparison["selection_metric"])
print("Best score:", f'{comparison["best_score"]:.4f}')