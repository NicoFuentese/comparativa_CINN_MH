import torch
import pandas as pd
import os
import random
import numpy as np

from src.data_loader import load_raw_data
from src.data_cleaner import clean_clinical_data, build_daily_tensors
from src.model import SchedulePINN
from src.trainer import train_model
from src.post_processing import extract_topology, simulated_annealing_optimization, hill_climbing_optimization
from src.visualization import plot_advanced_gantt, plot_wait_histograms, plot_convergence_curve

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    print("======================================================")
    print("  Hospital CINN Scheduling")
    print("======================================================")
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Dispositivo configurado: {device}")

    os.makedirs('data/processed', exist_ok=True)
    raw_csv_path = 'data/raw/2_dataset_procesado_actualizado.csv'

    #Cargar Datos
    print("\n[1/5] Extrayendo y curando datos hospitalarios...")
    df = load_raw_data(raw_csv_path)
    df_clean = clean_clinical_data(df)
    p_medical, p_occupancy, J, I, R = build_daily_tensors(df_clean, 
                                                        target_date="2023-02-01", 
                                                        num_samples=16, 
                                                        buffer_time=20.0, 
                                                        device=device)

    #Inicializar Modelo
    print("\n[2/5] Inicializando Red Neuronal CINN...")
    model = SchedulePINN(J, I, R).to(device)

    #Entrenamiento
    print("\n[3/5] Entrenando con Restricciones KKT...")
    model, s_pred, m_probs, training_history = train_model(model, p_medical, p_occupancy, J, I, R, device, MAX_STEPS=10000)

    # Post-Procesamiento (Hill Climbing)
    print("\n[4/5] Decodificando y aplicando Recocido Simulado...")
    task_data = extract_topology(s_pred, m_probs, p_medical, p_occupancy, J, I, R)

    #para usar SA
    best_tasks = simulated_annealing_optimization(task_data, J, iterations=5000)

    #para usar HC
    #best_tasks = hill_climbing_optimization(task_data, J, iterations=2000)

    # Guardado y Visualizar
    print("\n[5/5] Generando Reportes...")
    plot_convergence_curve(training_history)
    df_final = pd.DataFrame(best_tasks)
    df_final.to_csv("data/processed/solucion_final_optimizada.csv", index=False)
    
    makespan_final = df_final['real_end'].max()
    plot_advanced_gantt(df_final, makespan_final, J)
    plot_wait_histograms(df_final)

    print("\n======================================================")
    print(f"Makespan Final: {makespan_final:.2f} min.")
    print("  Archivos guardados en 'data/processed/'")
    print("======================================================")

if __name__ == "__main__":
    main()