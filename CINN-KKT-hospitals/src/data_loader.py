import pandas as pd

def load_raw_data(csv_path):
    """
    Lee el archivo CSV crudo desde el disco.
    No aplica ninguna limpieza ni regla clínica.
    """
    print(f"[*] Cargando dataset crudo desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"    -> Datos cargados exitosamente: {len(df)} filas originales.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No se encontró el archivo en {csv_path}")