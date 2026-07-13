# Reglas de comportamiento
Respuestas deben ser siempre en español.

# Tarea Principal
Este proyecto actualmente genera data sintetica con data_generator.py. Tu tarea es modificarlo para que utilice los datos de "real_surgeries_data.csv" que se encuentra en la carpera /data, y lea los datos de la fecha "2023-02-01", esto implementado en un archivo llamado data_loader.py en la carpeta /data. La implementación de las metaheuristicas esat bien, solamente hay que modificar el proyecto para que funcione con datos reales del archivo CSV. 

# Flujo de trabajo recomendado
1. Revisa el archivo real_surgeries_data.csv
2. Analiza el proyecto completo y comprende su funcionamiento y flujo de datos.
3. Implementa los cambios de la Tarea Principal

# Codigo de referencia
Te entrego una implementacion que se realizo para este proyecto, puedes replicarlo o modificarlo pero las reglas de negocio estan en este codigo:

```python
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Read raw CSV data."""
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo no existe: {csv_path}")
    
    print(f"[*] Cargando dataset crudo desde: {csv_path}")
    logger.info(f"Cargando dataset crudo desde: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Se cargaron {len(df)} filas del dataset")
    return df


def clean_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clinical validation rules."""
    required_columns = [
        'Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención',
        'Ingreso Recuperación', 'Salida Recuperación', 'Tiempo Intervención', 'Tiempo Recuperación'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el dataset: {missing_cols}")
    
    df_clean = df.dropna(subset=required_columns).copy()
    logger.info(f"Después de eliminar NaN: {len(df_clean)} filas")
    
    # Convert date columns
    date_columns = ['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 
                   'Ingreso Recuperación', 'Salida Recuperación']
    for col in date_columns:
        df_clean[col] = pd.to_datetime(df_clean[col])
    
    # Calculate durations
    df_clean['dur_pre'] = (df_clean['Inicio Intervención'] - df_clean['Ingreso Pabellón']).dt.total_seconds() / 60.0
    df_clean['dur_qx'] = df_clean['Tiempo Intervención'].astype(float)
    df_clean['dur_post'] = df_clean['Tiempo Recuperación'].astype(float)
    
    # Apply clinical filters
    initial_count = len(df_clean)
    df_clean = df_clean[
        (df_clean['dur_pre'] > 0) &
        (df_clean['dur_qx'] > 0) &
        (df_clean['dur_post'] >= 1.0) &
        (df_clean['dur_post'] <= 240.0)
    ]
    
    if initial_count - len(df_clean) > 0:
        logger.warning(f"Se filtraron {initial_count - len(df_clean)} registros por criterios clínicos")
    
    result = df_clean[['dur_pre', 'dur_qx', 'dur_post']].reset_index(drop=True)
    logger.info(f"{len(result)} pacientes sobrevivieron al filtro.")
    return result


def load_csv_surgeries_data(csv_path: str = None) -> dict:
    """
    Load surgical data and map to job IDs 1-15.
    
    Returns:
        dict: {job_id: {1: dur_pre, 2: dur_qx, 3: dur_post}, ...}
    """
    if csv_path is None:
        csv_path = "data/real_surgeries_data.csv"
    
    try:
        logger.info(f"Iniciando carga de datos de cirugía desde: {csv_path}")
        df_raw = load_raw_data(csv_path)
        df_clean = clean_clinical_data(df_raw)
        
        if len(df_clean) < 15:
            error_msg = f"Insuficientes filas. Se esperaban 15, se obtuvieron {len(df_clean)}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        df_sample = df_clean.head(15)
        
        result_dict = {}
        for job_id, (_, row) in enumerate(df_sample.iterrows(), start=1):
            result_dict[job_id] = {
                1: round(float(row['dur_pre']), 2),
                2: round(float(row['dur_qx']), 2),
                3: round(float(row['dur_post']), 2)
            }
        
        logger.info(f"Se cargaron {len(result_dict)} trabajos de cirugía exitosamente.")
        return result_dict
        
    except Exception as e:
        logger.error(f"Error al procesar datos de cirugía: {e}")
        raise
```