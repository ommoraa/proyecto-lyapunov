README — Proyecto IRAG–Lyapunov Stability Analyzer
## 1. Título del Proyecto

Análisis de Estabilidad y Predicción de Casos IRAG mediante Modelos Supervisados y Pipeline MLOps Integrado

## 2. Descripción General

Este proyecto desarrolla un sistema de análisis, modelación y predicción para el comportamiento de casos de Infección Respiratoria Aguda Grave (IRAG), incorporando principios de estabilidad dinámica y técnicas modernas de aprendizaje supervisado.

Se construye un pipeline completo de MLOps utilizando Git, DVC, MLflow y Streamlit, garantizando reproducibilidad, trazabilidad de experimentos y control de versiones de datos y modelos. El proyecto apoya procesos investigativos orientados a caracterizar la estabilidad epidemiológica mediante modelos supervisados de clasificación.

## 3. Estructura del Repositorio
proyecto-lyapunov/
├── data/
│   ├── raw/                 # Datos originales versionados con DVC
│   └── clean/               # Datos limpios listos para modelado
│
├── notebooks/
│   ├── 01_eda_and_data_understanding.ipynb
│   └── 02_model_experimentation.ipynb
│
├── src/
│   ├── clean_data.py
│   ├── prepare_data.py
│   └── train.py
│
├── models/
│   └── lyapunov_irag_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── reports/
│   └── informe_final.pdf
│
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md

## 4. Pipeline del Proyecto

El flujo completo está automatizado mediante DVC:

- clean_data
    Procesamiento inicial, validación, tratamiento de valores faltantes y depuración.

- prepare_data
    Selección y transformación de variables, normalización, codificación y división en entrenamiento y prueba.

- train_model
    Entrenamiento supervisado, registro de experimentos con MLflow, evaluación comparativa y selección del modelo con mejor desempeño.
    El modelo final se registra en la Model Registry como IRAG_Lyapunov_Champion.

- Ejecución del pipeline: dvc repro

## 5. Modelos Implementados
- Modelo Base: Logistic Regression

    Se utiliza como referencia inicial para establecer una línea base interpretativa.

- Modelo Mejorado: Random Forest

    Presenta el mejor rendimiento en métricas globales y suele ser seleccionado como modelo final del proyecto.

- Registro en MLflow

    Se registran automáticamente hiperparámetros, métricas, artefactos y el modelo serializado.
    El modelo con mayor f1-score se registra bajo el nombre: IRAG_Lyapunov_Champion

## 6. Reproducibilidad
- Versionamiento de datos
    dvc pull

- Ejecución completa del pipeline
    dvc repro

- Interfaz de trazabilidad de modelos
    mlflow ui

- Instalación del entorno
    pip install -r requirements.txt

## 7. Aplicación Streamlit

El proyecto incluye una aplicación interactiva que permite realizar predicciones y visualizar información relevante del modelo seleccionado.

- Ejecución: streamlit run app/streamlit_app.py

## 8. Consideraciones Metodológicas

El proyecto se enmarca en un enfoque riguroso orientado a la investigación científica.
Las decisiones de modelación, preprocesamiento y evaluación se justifican de acuerdo con:

- La estructura del dataset IRAG.

- La naturaleza del problema de clasificación.

- La pertinencia de métricas equilibradas en escenarios con clases desbalanceadas.

- La necesidad de estabilidad y trazabilidad en procesos experimentales.

## 9. Ejecución del Proyecto desde Cero
- git clone <https://github.com/ommoraa/proyecto-lyapunov.git>
- cd proyecto-lyapunov
- pip install -r requirements.txt
- dvc pull
- dvc repro
- mlflow ui

## 10. Limitaciones

- Variabilidad dependiente del tamaño y consistencia del dataset.

- Posible sobreajuste del modelo Random Forest con conjuntos reducidos.

- Requiere mayor profundización para integración matemática completa con funciones de estabilidad tipo Lyapunov.

## 11. Líneas Futuras de Trabajo

- Incorporación de técnicas de explicabilidad (SHAP).

- Evaluación de arquitecturas temporales (LSTM, Transformers).

- Extensión del modelo hacia criterios continuos de estabilidad dinámica.

- Despliegue en contenedores y automatización CI/CD.

## 12. Licencia

    Proyecto con fines académicos y de investigación.

## 13. Aplicación Web (Streamlit)

La implementación incluye una aplicación web construida en Streamlit, desplegada para acceso público en el enlace https://proyecto-lyapunov-ihomylr9mn3tyvnxvte779.streamlit.app/
    Esta herramienta permite interactuar con el modelo campeón registrado, generar predicciones a partir de nuevos datos, visualizar métricas de desempeño e interpretar resultados relevantes para el análisis de estabilidad epidemiológica. La aplicación constituye un componente fundamental para la transferencia de los resultados del proyecto, facilitando su uso por parte de investigadores, analistas y tomadores de decisión.

### Nota sobre la elaboración del proyecto
Este proyecto fue elaborado por los autores y contó con asistencia de ChatGPT en la corrección, depuración, estandarización y adaptación del código Python, así como en la organización de notebooks, scripts, documentación técnica y estructuración del pipeline. La responsabilidad final sobre el diseño, decisiones metodológicas y contenido analítico es completamente de los autores.

## Autores  

- Arsenio Hidalgo Troya  

- Oscar Mauricio Mora Arroyo  
  Asignatura: Ciencia de Datos para la Investigación Científica  
  Programa: Doctorado en Ciencias Naturales y Matemáticas  
  Universidad de Nariño (UDENAR) – 2025