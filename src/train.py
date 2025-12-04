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
    """
    Carga los archivos CSV de entrenamiento y prueba con validación básica.
    """
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
    """
    Separa las características (X) y la variable objetivo (y) a partir de la columna objetivo.
    """
    if target_col not in df.columns:
        print(f"[train][ERROR] La columna objetivo '{target_col}' no existe.")
        print(f"Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def compute_metrics(model, X_test, y_test):
    """
    Calcula las métricas de desempeño: accuracy, precision, recall, f1 y roc_auc (si hay predict_proba).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            # Por si en el conjunto de test solo hay una clase
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def log_run_to_mlflow(model, run_name: str, params: dict, metrics: dict) -> str:
    """
    Registra un experimento en MLflow (parámetros, métricas y modelo)
    y devuelve el run_id para poder registrar el modelo campeón después.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Parámetros
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Métricas
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Modelo como artifact del run
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        print(f"[mlflow] Run '{run_name}' registrado con run_id={run_id}")
        return run_id


def train_and_log_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Entrena LogisticRegression (baseline) y RandomForest (modelo mejorado),
    registra ambos en MLflow y devuelve el modelo con mejor F1-score.
    Además, registra el modelo campeón en la Model Registry de MLflow.
    """
    target_col = "target"

    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)

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

    print("[train] Entrenando modelo baseline (Logistic Regression)...")
    logreg.fit(X_train, y_train)
    logreg_metrics = compute_metrics(logreg, X_test, y_test)

    print("\n=== Resultados Logistic Regression (baseline) ===")
    for k, v in logreg_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    logreg_run_id = log_run_to_mlflow(
        logreg,
        "logreg_baseline",
        logreg_params,
        logreg_metrics,
    )

    # =========================================
    # 2. Modelo mejorado: Random Forest
    # =========================================
    rf_params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=rf_params["random_state"],
        n_jobs=rf_params["n_jobs"],
    )

    print("\n[train] Entrenando modelo mejorado (Random Forest)...")
    rf.fit(X_train, y_train)
    rf_metrics = compute_metrics(rf, X_test, y_test)

    print("\n=== Resultados Random Forest (mejorado) ===")
    for k, v in rf_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    rf_run_id = log_run_to_mlflow(
        rf,
        "random_forest_champion",
        rf_params,
        rf_metrics,
    )

    # ==================================================
    # Seleccionar modelo "campeón" usando F1-score
    # ==================================================
    f1_logreg = logreg_metrics["f1"]
    f1_rf = rf_metrics["f1"]

    if f1_rf >= f1_logreg:
        best_model = rf
        best_name = "Random Forest"
        champion_run_id = rf_run_id
    else:
        best_model = logreg
        best_name = "Logistic Regression"
        champion_run_id = logreg_run_id

    print(f"\n[train] Modelo seleccionado según F1-score: {best_name}")

    # ==================================================
    # Registrar el modelo campeón en la Model Registry
    # ==================================================
    try:
        model_uri = f"runs:/{champion_run_id}/model"
        registered_name = "IRAG_Lyapunov_Champion"
        mlflow.register_model(model_uri=model_uri, name=registered_name)
        print(
            f"[mlflow] Modelo campeón registrado en la Model Registry como "
            f"'{registered_name}' desde run_id={champion_run_id}"
        )
    except Exception as e:
        # No detiene el entrenamiento si falla el registro; solo avisa.
        print(f"[mlflow][WARN] No se pudo registrar el modelo campeón: {e}")

    return best_model


def main(train_path: str, test_path: str, model_output_path: str):
    """
    Función principal para entrenar el modelo y guardar el artefacto final.
    """
    # Configurar MLflow (tracking local en la carpeta del proyecto)
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("IRAG_Stability_Analyzer")

    # Cargar datos
    train_df, test_df = load_data(train_path, test_path)

    # Entrenar modelos y obtener el modelo seleccionado
    best_model = train_and_log_models(train_df, test_df)

    # Guardar modelo en una ruta fija (para la aplicación) y en la ruta parametrizada
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    fixed_model_path = "models/lyapunov_irag_model.pkl"

    dump(best_model, fixed_model_path)
    dump(best_model, model_output_path)

    print(f"[train] Modelo final guardado en: {fixed_model_path}")
    print(f"[train] Modelo final guardado en: {model_output_path}")


if __name__ == "__main__":
    """
    Uso esperado (por ejemplo, desde DVC):
        python src/train.py data/clean/train.csv data/clean/test.csv models/lyapunov_irag_model.pkl
    """
    if len(sys.argv) != 4:
        print(
            "Uso: python src/train.py <train_path> <test_path> <model_output_path>",
        )
        sys.exit(1)

    train_path_arg = sys.argv[1]
    test_path_arg = sys.argv[2]
    model_output_path_arg = sys.argv[3]

    main(train_path_arg, test_path_arg, model_output_path_arg)