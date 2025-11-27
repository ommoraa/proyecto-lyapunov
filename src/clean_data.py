import os
import sys
import pandas as pd


def clean_data(input_path: str, output_path: str) -> None:
    """
    Limpia y agrega la base de IRAG a nivel semana–año.

    - Valida que el archivo exista.
    - Renombra columnas clave (fecha, semana, año).
    - Convierte la fecha a datetime (por si se requiere después).
    - Elimina columnas no numéricas que no se usan en la agregación.
    - Convierte semana y año a numéricos.
    - Elimina filas sin semana o año.
    - Agrupa por (anio, semana) sumando únicamente columnas numéricas.
    - Guarda el resultado en output_path.
    """

    print("[clean_data] Leyendo archivo crudo:", input_path)

    # ------------------------------------------------------------
    # Validar archivo de entrada
    # ------------------------------------------------------------
    if not os.path.exists(input_path):
        print(f"[clean_data][ERROR] No existe el archivo: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # ------------------------------------------------------------
    # Renombrar columnas importantes
    # ------------------------------------------------------------
    df = df.rename(
        columns={
            "FECHA NOTIFICACION": "fecha",
            "SEMANA EPIDEMIOLOGICA": "semana",
            "AÑO": "anio",
        }
    )

    print("[clean_data] Convirtiendo fechas…")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # ------------------------------------------------------------
    # Columnas NO numéricas que no se usarán en la agregación
    # (se eliminan para evitar problemas al agrupar/sumar)
    # ------------------------------------------------------------
    cols_no_utiles = [
        "ORDEN",
        "UNIDAD PRIMARIA GENERADORA DEL DATO QUE NOTIFICA EL EVENTO",
        "FECHA NOTIFICACION",  # nombre original
        "fecha",               # versión convertida; no la usamos en la suma semanal
    ]

    df = df.drop(columns=cols_no_utiles, errors="ignore")

    # ------------------------------------------------------------
    # Asegurar tipos numéricos para semana y año
    # ------------------------------------------------------------
    df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")

    # Eliminar filas sin semana o año
    df = df.dropna(subset=["anio", "semana"])

    print("[clean_data] Agrupando por año–semana…")

    # ------------------------------------------------------------
    # Seleccionar SOLO columnas numéricas y agrupar
    # (incluye anio y semana porque ya son numéricas)
    # ------------------------------------------------------------
    df_numeric = df.select_dtypes(include=["number"])

    # groupby sobre anio y semana, sumando el resto de columnas numéricas
    df_weekly = df_numeric.groupby(["anio", "semana"], as_index=False).sum()

    # ------------------------------------------------------------
    # Crear carpeta de salida si no existe
    # ------------------------------------------------------------
    out_dir = os.path.dirname(output_path)
    if out_dir:  # evita error si output_path es solo nombre de archivo
        os.makedirs(out_dir, exist_ok=True)

    # Guardar
    df_weekly.to_csv(output_path, index=False)
    print(f"[clean_data] Archivo limpio guardado en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python clean_data.py <input_path> <output_path>")
        sys.exit(1)

    clean_data(sys.argv[1], sys.argv[2])