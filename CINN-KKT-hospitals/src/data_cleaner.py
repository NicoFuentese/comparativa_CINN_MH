import pandas as pd
import numpy as np
import torch

def clean_clinical_data(df):
    """
    Aplica las reglas de negocio clínico (Sanity Checks) y limpia datos nulos.
    """
    print("[*] Aplicando limpieza estricta y filtros clínicos...")
    
    cols_needed = ['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 
                   'Ingreso Recuperación', 'Salida Recuperación', 'Tiempo Intervención', 'Tiempo Recuperación']
    
    # 1. Filtro de completitud (Registros cerrados)
    df_clean = df.dropna(subset=cols_needed).copy()

    # 2. Conversión a fechas
    for col in ['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 'Ingreso Recuperación', 'Salida Recuperación']:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
    df_clean = df_clean.dropna(subset=['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 'Ingreso Recuperación', 'Salida Recuperación'])

    # 3. Calcular duraciones reales (en minutos)
    df_clean['dur_pre'] = (df_clean['Inicio Intervención'] - df_clean['Ingreso Pabellón']).dt.total_seconds() / 60.0
    df_clean['dur_qx'] = df_clean['Tiempo Intervención']
    df_clean['dur_post'] = df_clean['Tiempo Recuperación']

    # 4. Reglas Clínicas de Bai (Sanity Check)
    df_clean = df_clean[
        (df_clean['dur_pre'] > 0) & 
        (df_clean['dur_qx'] > 0) & 
        (df_clean['dur_post'] >= 1.0) &      # Mínimo 1 minuto
        (df_clean['dur_post'] <= 240.0)      # Máximo 4 horas
    ]
    
    print(f"    -> Datos limpios: {len(df_clean)} pacientes sobrevivieron al filtro.")
    return df_clean


def build_daily_tensors(df_clean, target_date, num_samples, buffer_time, device):
    """
    Filtra los datos por un día específico y los convierte en tensores para PyTorch.
    """
    print(f"[*] Extrayendo datos para el escenario: {target_date}")
    
    df_clean['date'] = df_clean['Ingreso Pabellón'].dt.date
    df_day = df_clean[df_clean['date'] == pd.to_datetime(target_date).date()]
    df_day = df_day.head(num_samples).reset_index(drop=True)

    if len(df_day) == 0:
        raise ValueError(f"No hay datos suficientes para la fecha {target_date}.")

    # Construcción de matrices
    p_time_medical_np = df_day[['dur_pre', 'dur_qx', 'dur_post']].values.astype(np.float32)
    J, I = p_time_medical_np.shape
    R = 4  # Capacidad fija de 4 pabellones por etapa

    p_time_medical = torch.tensor(p_time_medical_np, device=device)
    p_time_occupancy = p_time_medical + buffer_time

    print(f"    -> Tensores listos: {J} pacientes procesados en {I} etapas (R={R}).")
    return p_time_medical, p_time_occupancy, J, I, R