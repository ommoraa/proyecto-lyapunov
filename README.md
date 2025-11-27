# 📊 Proyecto IRAG Stability Analyzer

Proyecto IRAG Stability Analyzer

Este proyecto desarrolla un flujo completo para el análisis de datos relacionados con IRAG (Infecciones Respiratorias Agudas Graves), desde la limpieza y preparación de datos hasta la construcción de un modelo predictivo y una aplicación interactiva para su uso práctico.

El objetivo central es identificar semanas con comportamiento inusual a partir de datos históricos semanalizados.

## 📁 Estructura del proyecto

proyecto-lyapunov/
│
├── app/
│   └── irag_app.py              # Aplicación Streamlit para realizar predicciones
│
├── data/
│   ├── raw/                     # Archivos de datos originales
│   └── clean/                   # Archivos generados por el pipeline (train/test)
│
├── models/
│   └── lyapunov_irag_model.pkl  # Modelo final exportado para la aplicación
│
├── notebooks/
│   ├── 01_eda.ipynb             # Análisis exploratorio de datos
│   └── 02_model_experimentation.ipynb   # Modelación y experimentación
│
├── reports/
│   └── ...                      # Documentos, gráficos o informes adicionales
│
├── src/
│   ├── clean_data.py            # Limpieza y agregación de datos
│   ├── prepare_data.py          # Creación de variables derivadas y división train/test
│   └── train.py                 # Entrenamiento de modelos y registro con MLflow
│
├── dvc.yaml                     # Pipeline reproducible con DVC
├── dvc.lock                     # Estado del pipeline
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documento actual

---

## Flujo general de trabajo

1. **Limpieza y estructuración de datos:** conversión de fechas, selección de columnas relevantes, agregación semanal.  
2. **Preparación del dataset para modelado:** cálculo de proporciones, tasas de crecimiento y variable objetivo.  
3. **Modelación supervisada:** entrenamiento de modelos (Regresión Logística y Random Forest) y evaluación con métricas de clasificación.  
4. **Registro y trazabilidad de experimentos:** uso de MLflow para registrar parámetros, métricas y artefactos.  
5. **Aplicación interactiva:** app creada con Streamlit que permite cargar datos, ejecutar predicciones y descargar resultados.

---

## Nota sobre la elaboración del proyecto

Este proyecto fue elaborado por el autor y contó con asistencia de ChatGPT en la corrección, depuración, estandarización y adaptación del código Python, así como en la organización de notebooks, scripts, documentación técnica y estructuración del pipeline. La responsabilidad final sobre el diseño, decisiones metodológicas y contenido analítico es completamente del autor.
