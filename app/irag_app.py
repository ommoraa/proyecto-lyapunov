import joblib
from pathlib import Path
import pandas as pd
import streamlit as st

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(
    page_title="IRAG – Modelo de Predicción",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 IRAG – Analizador de Riesgo")
st.markdown(
    """
Esta aplicación carga el **modelo campeón** entrenado en el proyecto Lyapunov y 
permite hacer predicciones sobre nuevos datos.

### **Flujo de uso**
1. Carga el modelo desde `models/`.
2. Elige si quieres usar un dataset de ejemplo (`data/clean/test.csv`) o subir tu propio CSV.
3. La app calcula la predicción para cada fila (0 = bajo riesgo, 1 = alto riesgo) y muestra el resultado.
4. Puedes descargar el archivo con las predicciones.
"""
)


# =========================
# FUNCIONES AUXILIARES
# =========================

@st.cache_resource
def load_model(model_path: str):
    """
    Carga el modelo entrenado desde un archivo .pkl usando joblib.
    """
    path = Path(model_path)
    if not path.exists():
        st.error(
            f"No se encontró el archivo del modelo en: `{path}`\n\n"
            "Verifica el nombre en la carpeta `models/` "
            "y actualiza la constante MODEL_PATH en `irag_app.py`."
        )
        return None

    # Cargar el modelo con joblib (NO pickle)
    model = joblib.load(path)
    return model


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones mínimas iguales a las del entrenamiento.
    """
    df = df.copy()
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    # Si en train.py usaste FEATURES = [...], aquí debes replicarlo.
    # Por ahora dejamos todas las columnas.
    return df


def make_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el modelo sobre el DataFrame y añade:
    - prediccion
    - probabilidad (si está disponible)
    """
    features = prepare_features(df)

    y_pred = model.predict(features)

    result = df.copy()
    result["prediccion"] = y_pred

    if hasattr(model, "predict_proba"):
        result["probabilidad"] = model.predict_proba(features)[:, 1]

    return result


# =========================
# CARGA DEL MODELO
# =========================

MODEL_PATH = "models/lyapunov_irag_model.pkl"

st.sidebar.header("⚙️ Configuración")
st.sidebar.markdown("Ruta del modelo que se cargará:")
st.sidebar.code(MODEL_PATH, language="bash")

model = load_model(MODEL_PATH)

if model is None:
    st.stop()

st.success("✅ Modelo cargado correctamente.")


# =========================
# SELECCIÓN DE FUENTE DE DATOS
# =========================

st.header("📥 1. Cargar datos para predicción")

modo = st.radio(
    "Selecciona cómo quieres cargar los datos:",
    (
        "Usar dataset de ejemplo (data/clean/test.csv)",
        "Subir un archivo CSV propio",
    ),
)

df_input = None

if modo == "Usar dataset de ejemplo (data/clean/test.csv)":
    demo_path = Path("data/clean/test.csv")
    if not demo_path.exists():
        st.error(
            "No se encontró el archivo de ejemplo en `data/clean/test.csv`.\n\n"
            "Verifica que exista o usa la opción de subir tu propio CSV."
        )
    else:
        df_input = pd.read_csv(demo_path)
        st.info(f"Se cargó el dataset de ejemplo desde `{demo_path}`.")
        st.dataframe(df_input.head())
else:
    uploaded_file = st.file_uploader(
        "Sube un archivo CSV con la misma estructura que el conjunto de entrenamiento",
        type=["csv"],
    )
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente. Vista previa:")
        st.dataframe(df_input.head())


# =========================
# PREDICCIÓN
# =========================

st.header("🔮 2. Ejecutar predicción")

if df_input is not None:
    if st.button("Calcular predicciones"):
        try:
            df_result = make_predictions(model, df_input)
        except Exception as e:
            st.error(
                "Ocurrió un error al hacer las predicciones. "
                "Revisa que las columnas del CSV coincidan con las utilizadas en el entrenamiento."
            )
            st.exception(e)
        else:
            st.success("✅ Predicciones calculadas correctamente.")
            st.subheader("Resultado (primeras filas)")
            st.dataframe(df_result.head())

            # Métrica rápida: tasa de positivos
            if "prediccion" in df_result.columns:
                pos_rate = df_result["prediccion"].mean()
                st.metric(
                    "Proporción de casos positivos predichos",
                    f"{pos_rate:.2%}",
                )

            # Descargar CSV
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar CSV con predicciones",
                data=csv_out,
                file_name="predicciones_irag.csv",
                mime="text/csv",
            )
else:
    st.info("Carga un dataset de ejemplo o sube tu archivo CSV para habilitar las predicciones.")