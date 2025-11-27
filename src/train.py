import os
import sys
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from joblib import dump

import mlflow
import mlflow.sklearn


def load_data(train_path: str, test_path: str):
    """Carga los CSV de train y test con validación básica."""
    print(f"[train] Cargando train desde: {train_path}")
    print(f"[train] Cargando test  desde: {test_path}")

    if not os.path.exists(train_path):
        print(f"[train][ERROR] No existe el archivo: {train_path}")
        sys.exit(1)

    if not os.path.exists(test_path):
        print(f"[train][ERROR] No existe el archivo: {test_path}")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"[train] Shape train: {train_df.shape}")
    print(f"[train] Shape test : {test_df.shape}")

    return train_df, test_df


def split_features_target(df: pd.DataFrame, target_col: str = "target"):
    """Separa X (features) e y (target) a partir de la columna objetivo."""
    if target_col not in df.columns:
        print(f"[train][ERROR] La columna objetivo '{target_col}' no existe.")
        print(f"Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def compute_metrics(model, X_test, y_test):
    """Calcula accuracy, precision, recall, f1 y roc_auc (si hay predict_proba)."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # ROC-AUC solo si el modelo tiene predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def log_run_to_mlflow(model, run_name: str, params: dict, metrics: dict):
    """Registra un experimento en MLflow (parámetros, métricas y modelo)."""

    with mlflow.start_run(run_name=run_name):
        # parámetros
        for k, v in params.items():
            mlflow.log_param(k, v)

        # métricas
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # modelo
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[mlflow] Run '{run_name}' registrado.")


def train_and_log_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Entrena LogReg (baseline) y RandomForest (mejorado), registra ambos y devuelve el campeón."""

    TARGET_COL = "target"

    X_train, y_train = split_features_target(train_df, TARGET_COL)
    X_test, y_test = split_features_target(test_df, TARGET_COL)

    # ===========================
    # 1. Modelo baseline: LogReg
    # ===========================
    logreg_params = {
        "model_type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 500,
        "penalty": "l2",
        "solver": "lbfgs",
    }

    logreg = LogisticRegression(
        C=logreg_params["C"],
        max_iter=logreg_params["max_iter"],
        penalty=logreg_params["penalty"],
        solver=logreg_params["solver"],
    )

    print("[train] Entrenando modelo baseline (Logistic Regression)…")
    logreg.fit(X_train, y_train)
    logreg_metrics = compute_metrics(logreg, X_test, y_test)

    print("\n=== Resultados LogReg (baseline) ===")
    for k, v in logreg_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    log_run_to_mlflow(logreg, "logreg_baseline", logreg_params, logreg_metrics)

    # =========================================
    # 2. Modelo mejorado: Random Forest
    #    (a futuro se podría ajustar estos hiperparámetros si la intención es
    #     replicar exactamente el Notebook)
    # =========================================
    rf_params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    }

    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=rf_params["random_state"],
        n_jobs=-1,
    )

    print("\n[train] Entrenando modelo mejorado (Random Forest)…")
    rf.fit(X_train, y_train)
    rf_metrics = compute_metrics(rf, X_test, y_test)

    print("\n=== Resultados Random Forest (mejorado) ===")
    for k, v in rf_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    log_run_to_mlflow(rf, "random_forest_champion", rf_params, rf_metrics)

    # ==================================================
    # Seleccionar campeón (aquí usamos f1 como criterio)
    # ==================================================
    f1_logreg = logreg_metrics["f1"]
    f1_rf = rf_metrics["f1"]

    best_model = rf if f1_rf >= f1_logreg else logreg
    best_name = "Random Forest" if best_model is rf else "Logistic Regression"

    print(f"\n[train] Modelo campeón según F1-score: {best_name}")

    return best_model


def main(train_path: str, test_path: str, model_output_path: str):
    # Configurar MLflow
    mlflow.set_tracking_uri("mlruns")  # carpeta local del proyecto
    mlflow.set_experiment("IRAG_Stability_Analyzer")

    # Cargar datos
    train_df, test_df = load_data(train_path, test_path)

    # Entrenar modelos y obtener campeón
    best_model = train_and_log_models(train_df, test_df)
    import joblib
    joblib.dump(best_model, "models/lyapunov_irag_model.pkl")


    # Guardar modelo campeón para ser usado por la app
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    dump(best_model, model_output_path)
    print(f"[train] Modelo campeón guardado en: {model_output_path}")


if __name__ == "__main__":
    """
    Uso esperado (DVC ya lo hace así):
        python src/train.py data/clean/train.csv data/clean/test.csv models/lyapunov_irag_model.pkl
    """
    if len(sys.argv) != 4:
        print(
            "Uso: python src/train.py <train_path> <test_path> <model_output_path>",
        )
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_output_path = sys.argv[3]

    main(train_path, test_path, model_output_path)