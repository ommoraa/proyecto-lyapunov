import os
import sys
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from joblib import dump

import mlflow
import mlflow.sklearn


def load_data(train_path: str, test_path: str):
    print(f"[train] Cargando train desde {train_path}")
    print(f"[train] Cargando test  desde {test_path}")

    if not os.path.exists(train_path):
        print(f"[train][ERROR] No existe el archivo: {train_path}")
        sys.exit(1)

    if not os.path.exists(test_path):
        print(f"[train][ERROR] No existe el archivo: {test_path}")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Separa X (features) e y (target) a partir del nombre de la columna objetivo.
    """
    if target_col not in df.columns:
        print(f"[train][ERROR] La columna objetivo '{target_col}' no existe en el DataFrame.")
        print(f"Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, model_path: str):
    """
    Entrena un modelo de regresión logística y registra el experimento en MLflow.
    """

    # Separar X e y
    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)

    # Hiperparámetros (modelo "champion")
    C_value = 1.0
    max_iter = 500
    penalty = "l2"
    solver = "lbfgs"

    # Configurar MLflow (usa carpeta local mlruns/)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("IRAG_Stability_Analyzer")

    print("[train] Iniciando run en MLflow...")

    with mlflow.start_run(run_name="logreg_champion"):

        # Log de hiperparámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("solver", solver)
        mlflow.log_param("n_features", X_train.shape[1])

        # Definir y entrenar el modelo
        model = LogisticRegression(
            C=C_value,
            max_iter=max_iter,
            penalty=penalty,
            solver=solver,
        )
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        print(f"[train] accuracy = {acc:.4f}")
        print(f"[train] f1       = {f1:.4f}")
        print(f"[train] precision= {prec:.4f}")
        print(f"[train] recall   = {rec:.4f}")

        # Log de métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Guardar modelo en disco (para Streamlit)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        dump(model, model_path)
        print(f"[train] Modelo guardado en {model_path}")

    print("[train] Run de MLflow terminado.")


def main():
    # --- Leer argumentos que vienen desde dvc.yaml ---
    # cmd: python src/train.py data/clean/train.csv data/clean/test.csv models/lyapunov_irag_logreg.pkl
    if len(sys.argv) != 4:
        print("[train][ERROR] Uso: python src/train.py <train_path> <test_path> <model_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_path = sys.argv[3]

    # ⚠️ IMPORTANTE: La columna objetivo real en tu dataset se llama "target"
    TARGET_COLUMN = "target"

    train_df, test_df = load_data(train_path, test_path)
    train_model(train_df, test_df, TARGET_COLUMN, model_path)


if __name__ == "__main__":
    main()
