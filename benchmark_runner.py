import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path

# =============================================================================
# SETUP PATHS AND IMPORTS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
CINN_PATH = PROJECT_ROOT / "CINN-KKT-hospitals"
MH_PATH = PROJECT_ROOT / "MH-hospitals"

sys.path.append(str(CINN_PATH))
sys.path.append(str(MH_PATH))

# CINN Imports
try:
    from src.data_loader import load_raw_data as load_cinn_raw
    from src.data_cleaner import clean_clinical_data as clean_cinn_data, build_daily_tensors
    from src.model import SchedulePINN
    from src.trainer import train_model
    from src.post_processing import extract_topology, simulated_annealing_optimization
except ImportError as e:
    print(f"Error importing CINN modules: {e}")

# MH Imports
try:
    from config.config import get_algorithms, ALL_ROOMS
    from simulation.dynamic_scheduler import DynamicScheduler
except ImportError as e:
    print(f"Error importing MH modules: {e}")

# =============================================================================
# SHARED DATA LOADER
# =============================================================================

def get_shared_data(num_jobs, target_date="2023-02-01"):
    """
    Loads data for both models to ensure they use the exact same instances.
    """
    raw_csv = CINN_PATH / "data/raw/2_dataset_procesado_actualizado.csv"
    df = load_cinn_raw(str(raw_csv))
    df_clean = clean_cinn_data(df)
    
    # Filter by date
    df_clean['date_only'] = pd.to_datetime(df_clean['Ingreso Pabellón']).dt.strftime('%Y-%m-%d')
    day_df = df_clean[df_clean['date_only'] == target_date].head(num_jobs).copy()
    
    if len(day_df) < num_jobs:
        print(f"Warning: Requested {num_jobs} jobs but only {len(day_df)} available.")
    
    # Format for MH: {job_id: {1: dur_pre, 2: dur_qx, 3: dur_post}, ...}
    mh_data = {}
    for i, (_, row) in enumerate(day_df.iterrows(), start=1):
        mh_data[i] = {
            1: float(row['dur_pre']),
            2: float(row['dur_qx']),
            3: float(row['dur_post'])
        }
    
    return day_df, mh_data

# =============================================================================
# WRAPPERS
# =============================================================================

def run_cinn(num_jobs, seed, device='cpu'):
    """
    Wrapper for CINN model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Load Data
    raw_csv = CINN_PATH / "data/raw/2_dataset_procesado_actualizado.csv"
    df = load_cinn_raw(str(raw_csv))
    df_clean = clean_cinn_data(df)
    
    # 2. Build Tensors (using target_date and num_samples)
    p_medical, p_occupancy, J, I, R = build_daily_tensors(
        df_clean, target_date="2023-02-01", num_samples=num_jobs, buffer_time=20.0, device=device
    )
    
    t0 = time.perf_counter()
    
    # 3. Train PINN
    model = SchedulePINN(J, I, R).to(device)
    # Reducing MAX_STEPS slightly to speed up benchmark if needed, 
    # but maintaining quality for Q1.
    model, s_pred, m_probs, _ = train_model(model, p_medical, p_occupancy, J, I, R, device, MAX_STEPS=8000)
    
    # 4. Extract and Optimize (Post-processing)
    task_data = extract_topology(s_pred, m_probs, p_medical, p_occupancy, J, I, R)
    best_tasks = simulated_annealing_optimization(task_data, J, iterations=3000)
    
    elapsed = time.perf_counter() - t0
    
    df_res = pd.DataFrame(best_tasks)
    makespan = df_res['real_end'].max()
    
    return makespan, elapsed

def run_mh(algo_name, num_jobs, seed, mh_data):
    """
    Wrapper for Metaheuristics.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Patch JOB_TYPES to handle num_jobs
    from config.config import JOB_TYPES
    for i in range(1, num_jobs + 1):
        if i not in JOB_TYPES:
            # Assign a default type (e.g., 1) for jobs beyond 15
            JOB_TYPES[i] = (i % 3) + 1 
    
    # Find the specific algorithm spec
    all_algos = get_algorithms()
    spec = next((a for a in all_algos if a['name'].upper() == algo_name.upper() or a['name'] == algo_name), None)
    
    if spec is None:
        raise ValueError(f"Algorithm {algo_name} not found in MH project.")
    
    job_ids = list(range(1, num_jobs + 1))
    
    t0 = time.perf_counter()
    
    try:
        scheduler = DynamicScheduler(
            algorithm_runner=spec["runner"],
            surgeries_data=mh_data,
            job_ids=job_ids
        )
        # Using elective mode logic (no emergencies for benchmark)
        result = scheduler.run_with_emergencies([], seed=seed)
        schedule_details, events_log, makespan, best_hist, avg_hist = result
        
        elapsed = time.perf_counter() - t0
        return makespan, elapsed
    except Exception as e:
        print(f"Error running MH {algo_name}: {e}")
        return float('inf'), time.perf_counter() - t0

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def main():
    # As requested: Focusing on the 16 surgeries for that specific day
    INSTANCES = [16] 
    NUM_RUNS = 30
    MH_ALGOS = ['GA', 'dPSO', 'SBOA', 'dMShOA']
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Benchmark on {device} (Real Data: 2023-02-01, 16 surgeries)...")

    for num_jobs in INSTANCES:
        instance_name = f"RealDay_{num_jobs}Jobs"
        print(f"\n>>> Running Instance: {instance_name}")
        
        # Get shared data once per instance
        _, mh_data = get_shared_data(num_jobs)
        
        for run_i in range(NUM_RUNS):
            print(f"    Run {run_i+1}/{NUM_RUNS}...", end='\r')
            seed = run_i
            
            # 1. Run CINN
            makespan_cinn, time_cinn = run_cinn(num_jobs, seed, device=device)
            results.append({
                'Instance': instance_name,
                'Algorithm': 'CINN',
                'Run': run_i,
                'Makespan': makespan_cinn,
                'CPU_Time': time_cinn
            })
            
            # 2. Run MHs
            for algo in MH_ALGOS:
                makespan_mh, time_mh = run_mh(algo, num_jobs, seed, mh_data)
                results.append({
                    'Instance': instance_name,
                    'Algorithm': algo,
                    'Run': run_i,
                    'Makespan': makespan_mh,
                    'CPU_Time': time_mh
                })
    
    print("\nBenchmark Finished!")
    df_results = pd.DataFrame(results)
    df_results.to_csv("raw_results.csv", index=False)
    print("Saved results to raw_results.csv")

if __name__ == "__main__":
    main()
