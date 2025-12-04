# app/streamlit_app.py

import streamlit as st
import pandas as pd
from joblib import load
import os

st.set_page_config(page_title="IRAG - Detección de semanas inusuales", layout="centered")

st.title("IRAG - Detección de comportamiento epidémico inusual")
st.write(
    """
Esta aplicación utiliza el **modelo campeón** entrenado en el proyecto
para predecir si una semana epidemiológica presenta un comportamiento
**normal (0)** o **inusual (1)** de IRAG, a partir de indicadores agregados.
"""
)

MODEL_PATH = "models/lyapunov_irag_model.pkl"
TEST_PATH = "data/clean/test.csv"
TARGET_COL = "target"


@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo en: {model_path}")
        return None
    model = load(model_path)
    return model


@st.cache_data
def load_test_data(test_path: str):
    if not os.path.exists(test_path):
        st.error(f"No se encontró el archivo de test en: {test_path}")
        return None, None

    df = pd.read_csv(test_path)
    if TARGET_COL not in df.columns:
        st.error(f"No se encontró la columna target='{TARGET_COL}' en el test.")
        return None, None

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


model = load_model(MODEL_PATH)
X_test, y_test = load_test_data(TEST_PATH)

if (model is None) or (X_test is None):
    st.stop()

st.subheader("Selección de semana de prueba")

idx = st.number_input(
    "Seleccione el índice de una semana del conjunto de test:",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
)

sample_features = X_test.iloc[[idx]]  # mantiene DataFrame de una fila
st.write("Características de la semana seleccionada:")
st.dataframe(sample_features)

if st.button("Predecir comportamiento"):
    pred = model.predict(sample_features)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample_features)[0, 1]
    else:
        prob = None

    label = "INUSUAL (1)" if pred == 1 else "NORMAL (0)"
    st.markdown(f"### Predicción del modelo: **{label}**")

    if prob is not None:
        st.write(f"Probabilidad estimada de semana inusual: **{prob:.3f}**")


st.subheader("Importancia de variables (si aplica)")

import matplotlib.pyplot as plt

if hasattr(model, "feature_importances_"):
    # Intentar recuperar nombres de variables
    if hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_
    else:
        feature_names = X_test.columns

    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    st.write("Top 10 variables más importantes según el modelo Random Forest:")
    st.dataframe(fi_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    fi_df.sort_values("importance", ascending=True).plot(
        kind="barh",
        x="feature",
        y="importance",
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("Variable")
    ax.set_title("Top 10 variables más importantes")
    st.pyplot(fig)
else:
    st.info(
        "El modelo actual no expone atributo 'feature_importances_'. "
        "Esta sección solo aplica para modelos basados en árboles (Random Forest, etc.)."
    )