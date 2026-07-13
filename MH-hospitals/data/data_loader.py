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
        # Try relative to project root if not found
        project_root = Path(__file__).resolve().parent.parent
        file_path = project_root / csv_path
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {csv_path}")
    
    logger.info(f"Cargando dataset crudo desde: {file_path}")
    
    # Common encodings for Spanish CSVs: 'latin-1', 'utf-8', 'utf-8-sig'
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')
        
    logger.info(f"Se cargaron {len(df)} filas del dataset")
    return df

def clean_clinical_data(df: pd.DataFrame, target_date: str = "2023-02-01") -> pd.DataFrame:
    """Apply clinical validation rules and filter by date."""
    required_columns = [
        'Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención',
        'Ingreso Recuperación', 'Salida Recuperación', 'Tiempo Intervención', 'Tiempo Recuperación'
    ]
    
    # Handle potential encoding issues in column names if necessary
    # (Mapping known corrupted names to correct ones)
    column_mapping = {
        'Ingreso PabellÃ³n': 'Ingreso Pabellón',
        'Inicio IntervenciÃ³n': 'Inicio Intervención',
        'TÃ©rmino IntervenciÃ³n': 'Término Intervención',
        'Ingreso RecuperaciÃ³n': 'Ingreso Recuperación',
        'Salida RecuperaciÃ³n': 'Salida Recuperación',
        'Tiempo IntervenciÃ³n': 'Tiempo Intervención',
        'Tiempo RecuperaciÃ³n': 'Tiempo Recuperación'
    }
    df = df.rename(columns=column_mapping)
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el dataset: {missing_cols}")
    
    # Convert date columns to datetime
    date_columns = ['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 
                   'Ingreso Recuperación', 'Salida Recuperación']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Filter by date BEFORE dropping NaNs to be more efficient if file is huge
    # We use 'Ingreso Pabellón' for the date filter
    df['date_only'] = df['Ingreso Pabellón'].dt.strftime('%Y-%m-%d')
    df_filtered = df[df['date_only'] == target_date].copy()
    
    logger.info(f"Filas encontradas para la fecha {target_date}: {len(df_filtered)}")
    
    # Drop rows with NaNs in required columns
    df_clean = df_filtered.dropna(subset=required_columns).copy()
    logger.info(f"Después de eliminar NaN: {len(df_clean)} filas")
    
    # Calculate durations in minutes
    df_clean['dur_pre'] = (df_clean['Inicio Intervención'] - df_clean['Ingreso Pabellón']).dt.total_seconds() / 60.0
    df_clean['dur_qx'] = pd.to_numeric(df_clean['Tiempo Intervención'], errors='coerce')
    df_clean['dur_post'] = pd.to_numeric(df_clean['Tiempo Recuperación'], errors='coerce')
    
    # Re-drop NaNs if to_numeric failed for some rows
    df_clean = df_clean.dropna(subset=['dur_pre', 'dur_qx', 'dur_post'])
    
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
    logger.info(f"{len(result)} pacientes sobrevivieron al filtro clínico para la fecha {target_date}.")
    return result

def load_csv_surgeries_data(csv_path: str = "data/real_surgeries_data.csv", target_date: str = "2023-02-01") -> dict:
    """
    Load surgical data from CSV, filter by date, and map to job IDs.
    
    Returns:
        dict: {job_id: {1: dur_pre, 2: dur_qx, 3: dur_post}, ...}
    """
    try:
        logger.info(f"Iniciando carga de datos de cirugía para la fecha {target_date} desde: {csv_path}")
        df_raw = load_raw_data(csv_path)
        df_clean = clean_clinical_data(df_raw, target_date)
        
        if len(df_clean) == 0:
            logger.warning(f"No se encontraron datos válidos para la fecha {target_date}.")
            return {}
            
        result_dict = {}
        # Map all available rows from that date
        for job_id, (_, row) in enumerate(df_clean.iterrows(), start=1):
            result_dict[job_id] = {
                1: round(float(row['dur_pre']), 2),
                2: round(float(row['dur_qx']), 2),
                3: round(float(row['dur_post']), 2)
            }
        
        logger.info(f"Se cargaron {len(result_dict)} trabajos de cirugía exitosamente para el día {target_date}.")
        return result_dict
        
    except Exception as e:
        logger.error(f"Error al procesar datos de cirugía: {e}")
        raise
