import pandas as pd
import sys
import os
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow

def train(train_path, test_path, model_path):
    print("[train_model] Cargando datos…")

    if not os.path.exists(train_path):
        print("[train_model][ERROR] No existe train.csv")
        sys.exit(1)

    train_df = pd.read_csv(train_path)

    if train_df.empty:
        print("[train_model][ERROR] Train vacío. No se puede entrenar.")
        sys.exit(1)

    if len(train_df) < 5:
        print("[train_model] Muy pocos datos para entrenar. Creando modelo vacío.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(None, f)
        return

    # Variables de entrada
    X_train = train_df[[
        "I_t_minus_1",
        "growth_rate",
        "prop_hosp_irag",
        "prop_uci_irag",
        "prop_muertes_irag",
        "prop_consultas_irag",
    ]]

    y_train = train_df["target"]

    print("[train_model] Entrenando modelo…")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    print("[train_model] Guardando modelo en:", model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("[train_model] Listo.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python train.py <train_path> <test_path> <model_path>")
        sys.exit(1)

    train(sys.argv[1], sys.argv[2], sys.argv[3])
