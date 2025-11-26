import pandas as pd
import sys
import os

def clean_data(input_path, output_path):
    print("[clean_data] Leyendo archivo crudo:", input_path)

    # Validar archivo de entrada
    if not os.path.exists(input_path):
        print(f"[clean_data][ERROR] No existe el archivo: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Renombrar columnas importantes
    df = df.rename(columns={
        "FECHA NOTIFICACION": "fecha",
        "SEMANA EPIDEMIOLOGICA": "semana",
        "AÑO": "anio"
    })

    print("[clean_data] Convirtiendo fechas…")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Columnas no numéricas que se eliminarán antes del groupby
    cols_no_utiles = [
        "ORDEN",
        "UNIDAD PRIMARIA GENERADORA DEL DATO QUE NOTIFICA EL EVENTO",
        "FECHA NOTIFICACION",  # vieja
        "fecha",               # convertida pero no se usa en agregación
    ]

    df = df.drop(columns=cols_no_utiles, errors="ignore")

    # Convertir semana y año a números
    df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")

    # Eliminar filas sin semana o año
    df = df.dropna(subset=["anio", "semana"])

    print("[clean_data] Agrupando por año–semana…")

    # Agrupación SOLO con columnas numéricas
    df_numeric = df.select_dtypes(include=["number"])
    df_weekly = df_numeric.groupby(["anio", "semana"], as_index=False).sum()

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_weekly.to_csv(output_path, index=False)
    print(f"[clean_data] Archivo limpio guardado en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python clean_data.py <input_path> <output_path>")
        sys.exit(1)

    clean_data(sys.argv[1], sys.argv[2])
