import pandas as pd
import sys
import os


def prepare_data(clean_path: str, train_path: str, test_path: str) -> None:
    """
    Prepara el conjunto de datos para entrenamiento y prueba.

    Pasos principales:
    - Validación del archivo limpio.
    - Cálculo de proporciones relacionadas con IRAG.
    - Generación de variables derivadas (rezagos y tasas de crecimiento).
    - Definición de la variable objetivo.
    - División en subconjuntos de entrenamiento y prueba.
    - Guardado de los archivos resultantes.
    """

    print("[prepare_data] Leyendo archivo limpio:", clean_path)

    if not os.path.exists(clean_path):
        print("[prepare_data][ERROR] No existe:", clean_path)
        sys.exit(1)

    df = pd.read_csv(clean_path)

    if df.empty:
        print("[prepare_data][ERROR] El archivo está vacío.")
        sys.exit(1)

    print("[prepare_data] Generando variables de tasa...")

    # Proporciones
    df["prop_hosp_irag"] = df["TOTAL CASOS DE HOSPITALIZACIONES POR IRAG"] / (
        df["TOTAL HOSPITALIZACIONES POR TODAS LAS CAUSAS"] + 1
    )
    df["prop_uci_irag"] = df["TOTAL DE LAS HOSPITALIZACIONES EN UCI POR IRAG"] / (
        df["TOTAL TODAS LAS HOSPITALIZACIONES EN UCI POR TODAS LAS CAUSAS"] + 1
    )
    df["prop_muertes_irag"] = df["TOTAL DE MUERTES POR IRAG"] / (
        df["TOTAL DE MUERTES POR TODAS LAS CAUSAS"] + 1
    )
    df["prop_consultas_irag"] = (
        df["TOTAL DE EVENTOS DE MORBILIDAD DE IRAG POR CONSULTA EXTERNA Y URGENCIAS"]
        / (
            df[
                "TOTAL DE EVENTOS DE MORBILIDAD DE IRAG POR CONSULTA EXTERNA Y URGENCIAS POR TODAS LAS CAUSAS INCLUYENDO EL IRAG"
            ]
            + 1
        )
    )

    # Rezago de una semana
    df["I_t_minus_1"] = df["prop_hosp_irag"].shift(1)

    # Tasa de crecimiento
    df["growth_rate"] = (
        df["prop_hosp_irag"] - df["I_t_minus_1"]
    ) / (df["I_t_minus_1"] + 1)

    # Eliminar filas sin rezago
    df = df.dropna()

    # Definir variable objetivo
    df["target"] = (df["growth_rate"] > 0).astype(int)

    print("[prepare_data] División en entrenamiento y prueba...")
    n = len(df)

    if n < 20:
        print("[prepare_data] Cantidad insuficiente de datos. Todo va a entrenamiento.")
        df_train = df.copy()
        df_test = df.iloc[0:0]  # conjunto vacío
    else:
        split = int(n * 0.8)
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[prepare_data] Archivo de entrenamiento guardado en: {train_path}")
    print(f"[prepare_data] Archivo de prueba guardado en: {test_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python prepare_data.py <clean_path> <train_path> <test_path>")
        sys.exit(1)

    prepare_data(sys.argv[1], sys.argv[2], sys.argv[3])