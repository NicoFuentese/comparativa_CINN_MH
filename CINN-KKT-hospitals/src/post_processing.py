import numpy as np
import copy
import math
import random
import torch

def extract_topology(s_pred, m_probs, p_time_medical, p_time_occupancy, J, I, R):
    s_pred_np = s_pred.cpu().numpy()
    p_med_np = p_time_medical.cpu().numpy()
    p_occ_np = p_time_occupancy.cpu().numpy()
    machine_indices = torch.argmax(m_probs, dim=-1).cpu().numpy()

    task_data = []
    for j in range(J):
        for i in range(I):
            local_m = machine_indices[j, i]
            global_m_id = (i * R) + local_m + 1 
            stage_name = ["PRE", "QX", "POST"][i]
            res_name = f"{stage_name}-{local_m+1}"
            
            task_data.append({
                'job_id': j,
                'stage_id': i,
                'pinn_start': s_pred_np[j, i],
                'dur_medical': p_med_np[j, i],
                'dur_occupancy': p_occ_np[j, i],
                'global_machine_id': global_m_id,
                'resource_name': res_name
            })
    return task_data

# --- MODIFICADO: Ahora calcula Makespan, Esperas Y Desbalance ---
def calculate_metrics(tasks, J):
    machine_avail = {r: 0.0 for r in range(1, 13)} 
    machine_work_time = {r: 0.0 for r in range(1, 13)} # Cuánto trabaja cada sala
    job_avail = {j: 0.0 for j in range(J)}
    tasks.sort(key=lambda x: x['pinn_start'])
    
    final_tasks = []
    total_wait_time = 0.0 
    
    for t in tasks:
        j = t['job_id']
        m_global = t['global_machine_id']
        dur_med = t['dur_medical']
        dur_occ = t['dur_occupancy']
        
        start_t = max(job_avail[j], machine_avail[m_global])
        
        if t['stage_id'] > 0: 
            total_wait_time += (start_t - job_avail[j])
            
        end_t_patient = start_t + dur_med
        end_t_machine = start_t + dur_occ
        
        job_avail[j] = end_t_patient
        machine_avail[m_global] = end_t_machine
        machine_work_time[m_global] += dur_med # Acumulamos el trabajo de esta sala
        
        new_t = t.copy()
        new_t['real_start'] = start_t
        new_t['real_end'] = end_t_patient
        new_t['pinn_start'] = start_t 
        final_tasks.append(new_t)
        
    makespan = max([t['real_end'] for t in final_tasks])
    
    # Calcular el desbalance (Desviación estándar del tiempo trabajado por etapa)
    desbalance_total = 0.0
    for stage in range(3):
        cargas_etapa = [machine_work_time[m] for m in range((stage*4)+1, (stage*4)+5)]
        desbalance_total += np.std(cargas_etapa)
        
    return makespan, total_wait_time, desbalance_total, final_tasks

# --- MODIFICADO: Metaheurística SA Multi-Objetivo con Balanceo ---
def simulated_annealing_optimization(task_data, J, iterations=5000, initial_temp=100.0, cooling_rate=0.995):
    print("Iniciando Recocido Simulado Multi-Objetivo Avanzado (SA)...")
    best_tasks = copy.deepcopy(task_data)
    current_tasks = copy.deepcopy(task_data)
    
    best_makespan, best_waits, best_imb, best_tasks = calculate_metrics(best_tasks, J)
    
    # PESOS DE IMPORTANCIA:
    peso_espera = 0.15   # Cuida al paciente sin destruir el Makespan
    peso_balance = 0.50  # Fuerza a que todos los pabellones trabajen por igual
    
    best_cost = best_makespan + (peso_espera * best_waits) + (peso_balance * best_imb)
    current_cost = best_cost
    
    print(f"Inicio SA -> Makespan: {best_makespan:.1f} | Espera: {best_waits:.1f} | Desbalance: {best_imb:.1f}")
    
    machine_ranges = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11, 12]}
    T = initial_temp
    
    for k in range(iterations):
        neighbor_tasks = copy.deepcopy(current_tasks)
        idx = np.random.randint(0, len(neighbor_tasks))
        
        task = neighbor_tasks[idx]
        stage = task['stage_id']
        
        candidates = [m for m in machine_ranges[stage] if m != task['global_machine_id']]
        if not candidates: continue
            
        new_machine = np.random.choice(candidates)
        neighbor_tasks[idx]['global_machine_id'] = new_machine
        
        stage_name = ["PRE", "QX", "POST"][stage]
        neighbor_tasks[idx]['resource_name'] = f"{stage_name}-{new_machine - (stage * 4)}"
        
        neighbor_makespan, neighbor_waits, neighbor_imb, updated_schedule = calculate_metrics(neighbor_tasks, J)
        neighbor_cost = neighbor_makespan + (peso_espera * neighbor_waits) + (peso_balance * neighbor_imb)
        
        delta = neighbor_cost - current_cost
        
        if delta < 0 or math.exp(-delta / T) > random.random():
            current_tasks = updated_schedule
            current_cost = neighbor_cost
            
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                best_makespan = neighbor_makespan
                best_waits = neighbor_waits
                best_imb = neighbor_imb
                best_tasks = copy.deepcopy(current_tasks)
                
        T = T * cooling_rate

    print(f"Final SA -> Makespan: {best_makespan:.1f} | Espera: {best_waits:.1f} | Desbalance: {best_imb:.1f}")
    return best_tasks

# --- Algoritmo Hill Climbing (Actualizado para compatibilidad) ---
def hill_climbing_optimization(task_data, J, iterations=2000):
    print("Iniciando Búsqueda Local (Hill Climbing)...")
    best_tasks = copy.deepcopy(task_data)
    current_tasks = copy.deepcopy(task_data)
    
    best_makespan, best_waits, best_imb, best_tasks = calculate_metrics(best_tasks, J)
    
    peso_espera = 0.15
    peso_balance = 0.50
    best_cost = best_makespan + (peso_espera * best_waits) + (peso_balance * best_imb)
    current_cost = best_cost
    
    machine_ranges = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11, 12]}
    
    for k in range(iterations):
        neighbor_tasks = copy.deepcopy(current_tasks)
        idx = np.random.randint(0, len(neighbor_tasks))
        
        task = neighbor_tasks[idx]
        stage = task['stage_id']
        
        candidates = [m for m in machine_ranges[stage] if m != task['global_machine_id']]
        if not candidates: continue
            
        new_machine = np.random.choice(candidates)
        neighbor_tasks[idx]['global_machine_id'] = new_machine
        
        stage_name = ["PRE", "QX", "POST"][stage]
        neighbor_tasks[idx]['resource_name'] = f"{stage_name}-{new_machine - (stage * 4)}"
        
        neighbor_makespan, neighbor_waits, neighbor_imb, updated_schedule = calculate_metrics(neighbor_tasks, J)
        neighbor_cost = neighbor_makespan + (peso_espera * neighbor_waits) + (peso_balance * neighbor_imb)
        
        if neighbor_cost <= current_cost:
            current_tasks = updated_schedule
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_makespan = neighbor_makespan
                best_waits = neighbor_waits
                best_tasks = copy.deepcopy(current_tasks)

    return best_tasks