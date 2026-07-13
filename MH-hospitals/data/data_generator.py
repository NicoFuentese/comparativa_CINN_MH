import copy
import numpy as np
from data.data_loader import load_csv_surgeries_data
from config.config import NUM_SURGERIES

# Global cache to avoid re-loading the CSV in every simulation
_REAL_DATA_CACHE = None

def get_real_base_data():
    """
    Lazy loader for real surgery data.
    """
    global _REAL_DATA_CACHE
    if _REAL_DATA_CACHE is None:
        try:
            # We load valid surgeries for the specific date
            all_real_data = load_csv_surgeries_data(target_date="2023-02-01")
            
            # Slice according to NUM_SURGERIES defined in config
            sorted_keys = sorted(all_real_data.keys())
            _REAL_DATA_CACHE = {i+1: all_real_data[sorted_keys[i]] for i in range(min(NUM_SURGERIES, len(sorted_keys)))}
            
            if len(_REAL_DATA_CACHE) < NUM_SURGERIES:
                print(f"WARNING: Requested {NUM_SURGERIES} surgeries, but only {len(_REAL_DATA_CACHE)} available.")
                
        except Exception as e:
            print(f"Error loading real data, falling back to synthetic: {e}")
            # Fallback (keeping old synthetic data just in case)
            _REAL_DATA_CACHE = {
                i: {1: 30, 2: 60, 3: 40} for i in range(1, NUM_SURGERIES + 1)
            }
    return _REAL_DATA_CACHE

# Expose for backward compatibility
BASE_DAY_SURGERIES_DATA = get_real_base_data()

def generate_day_surgeries_data(job_ids, std_factor=0.0):
    """
    Generates surgery processing-time data using real data from CSV.
    Optional variability can be added via std_factor.
    """
    base_data = BASE_DAY_SURGERIES_DATA
    data = {}
    for j in job_ids:
        if j in base_data:
            data[j] = copy.deepcopy(base_data[j])
            
            if std_factor > 0:
                for op in [1, 2, 3]:
                    base_val = base_data[j][op]
                    std_val = std_factor * base_val
                    value = np.random.normal(base_val, std_val)
                    data[j][op] = max(1, round(value, 2))
        else:
            # Default value for undefined jobs (if any)
            data[j] = {1: 30, 2: 60, 3: 40}
    return data