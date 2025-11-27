import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import plotly.express as px


# ---------------------------------------------
# 1. Cargar modelo actual
# ---------------------------------------------
MODEL_PATH = "models/lyapunov_irag_logreg.pkl"

@st.cache_data
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()


# ---------------------------------------------
# 2. Cargar datos limpios para gráficas
# ---------------------------------------------
DATA_PATH = "data/clean/irag_clean.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

df = load_data()



# ---------------------------------------------
# App layout
# ---------------------------------------------
st.set_page_config(
    page_title="IRAG Stability Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 IRAG Stability Analyzer")
st.write("Aplicación avanzada para análisis epidemiológico con modelos predictivos y señales tipo Lyapunov")


# ======================================================
# Sidebar
# ======================================================
st.sidebar.header(" Navegación ")
pagina = st.sidebar.radio(
    "Selecciona sección:",
    ["🔮 Predicción IRAG", "📊 Gráficas", "📘 Explicación", "ℹ Sobre el Proyecto"]
)



# =================================================================
#  PAGINA 1 - Predicción
# =================================================================
if pagina == "🔮 Predicción IRAG":
    st.header("🔮 Predicción del comportamiento semanal del IRAG")

    if model is None:
        st.error("❌ Modelo no encontrado. Ejecuta `dvc repro`.")
        st.stop()

    st.subheader("✨ Ingrese los valores para predecir")

    col1, col2 = st.columns(2)

    with col1:
        I_t_minus_1 = st.number_input(
            "Proporción IRAG en semana anterior (Iₜ₋₁)",
            min_value=0.0, max_value=1.0, value=0.05
        )
        growth_rate = st.number_input(
            "Tasa de crecimiento (aprox. derivada discreta)",
            min_value=-1.0, max_value=1.0, value=0.02
        )
        prop_hosp_irag = st.number_input(
            "Proporción de hospitalizaciones IRAG",
            min_value=0.0, max_value=1.0, value=0.03
        )

    with col2:
        prop_uci_irag = st.number_input(
            "Proporción de UCI IRAG",
            min_value=0.0, max_value=1.0, value=0.01
        )
        prop_muertes_irag = st.number_input(
            "Proporción de muertes IRAG",
            min_value=0.0, max_value=1.0, value=0.01
        )
        prop_consultas_irag = st.number_input(
            "Proporción consultas IRAG",
            min_value=0.0, max_value=1.0, value=0.1
        )

    if st.button("🔍 Predecir tendencia"):
        X = np.array([[
            I_t_minus_1,
            growth_rate,
            prop_hosp_irag,
            prop_uci_irag,
            prop_muertes_irag,
            prop_consultas_irag
        ]])

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        st.subheader("📌 Resultado de predicción")

        if prob < 0.3:
            st.success(f"📉 IRAG en **tendencia estable o bajando** (prob={prob:.2f})")
        elif prob < 0.6:
            st.warning(f"⚠️ IRAG con **variabilidad moderada** (prob={prob:.2f})")
        else:
            st.error(f"🚨 IRAG en **riesgo de crecimiento** (prob={prob:.2f})")

        st.metric("Probabilidad de incremento", f"{prob:.2f}")



# =================================================================
#  PAGINA 2 - Gráficas
# =================================================================
elif pagina == "📊 Gráficas":
    st.header("📊 Series temporales y comportamiento del IRAG")

    if df is None:
        st.error("No se encontró el archivo limpio. Ejecute `dvc repro`.")
        st.stop()

    colA, colB = st.columns(2)

    with colA:
        fig = px.line(df, x="semana", y="TOTAL CASOS DE HOSPITALIZACIONES POR IRAG",
                      title="Hospitalizaciones IRAG por semana")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.line(df, x="semana", y="prop_hosp_irag",
                       title="Proporción IRAG en hospitalizaciones")
        st.plotly_chart(fig2, use_container_width=True)



# =================================================================
#  PAGINA 3 — Explicación
# =================================================================
elif pagina == "📘 Explicación":
    st.header("📘 Interpretación epidemiológica del modelo")

    st.write("""
    ### 🔹 Relación con estabilidad y funciones tipo Lyapunov
    - En epidemiología, observar si `prop_hosp_irag(t)` tiende a estabilizarse es equivalente 
      a estudiar la **estabilidad de un equilibrio**.
    - La variable `growth_rate` que usamos es una aproximación discreta de la **derivada** del sistema.
    - Si `growth_rate → 0` y `I_t_minus_1` se estabiliza → el sistema se aproxima a un **punto estable**.
    - Si `growth_rate > 0` persistente → indica **inestabilidad**, típico de brotes o ondas epidémicas.

    ### 🔹 ¿Cómo lo usa el modelo?
    - El modelo aprende patrones de transición semana a semana.
    - Cuando la probabilidad de incremento es alta, estamos en un régimen **inestable**.
    - Cuando es baja, el sistema converge a un equilibrio epidemiológico.

    """)



# =================================================================
#  PAGINA 4 — Sobre el Proyecto
# =================================================================
elif pagina == "ℹ Sobre el Proyecto":
    st.header("ℹ Información del Proyecto")
    st.write("**Proyecto Final de Ciencia de Datos — Lyapunov + IRAG**")
    st.write("**Autor:** Oscar Mauricio Mora Arroyo")
    st.write(f"📅 Última actualización: {datetime.date.today()}")

    st.write("🔗 *Enlace al repositorio GitHub (añádelo en el README cuando lo tengas listo: en unos momentos)*")
