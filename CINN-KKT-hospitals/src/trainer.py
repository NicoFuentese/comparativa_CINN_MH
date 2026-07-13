import torch
import torch.nn.functional as F
import copy
from src.constraints import DualVariables, build_constraints_v2, calculate_makespan

#entrenamiento
def train_model(model, p_time_medical, p_time_occupancy, J, I, R, device, MAX_STEPS=10000):
    job_ids = torch.arange(J, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    rho = 100.0          
    rho_max = 300000.0   
    rho_step = 100      
    beta = 1.2          
    tau_start = 2.0
    tau_end = 0.05       
    w_balance = 50.0 
    
    best_viol = float('inf')
    best_makespan = float('inf')
    best_model_state = None
    duals = None
    
    print("Iniciando Entrenamiento CINN...")
    
    training_history = []
    for step in range(MAX_STEPS):
        tau = max(tau_end, tau_start - (tau_start - tau_end) * (step / (MAX_STEPS * 0.9)))
        
        s_pred, m_probs = model(job_ids, tau=tau)
        g_val = build_constraints_v2(s_pred, p_time_medical, p_time_occupancy, m_probs)
        
        if step == 0: duals = DualVariables(g_val.shape[0], device)
        
        obj = calculate_makespan(s_pred, p_time_medical)
        
        # balanceo
        p_occ_expanded = p_time_occupancy.unsqueeze(-1)
        machine_loads = (m_probs * p_occ_expanded).sum(dim=0)
        load_std = machine_loads.std(dim=1).mean()
        
        lagrange_term = torch.dot(duals.mu, g_val) 
        penalty_term = 0.5 * rho * torch.sum(F.relu(g_val) ** 2)
        max_viol = torch.max(g_val).item()
        
        #Transición Sigmoidal / Exponencial del Peso del Objetivo
        # Si max_viol es 0, exp(0) = 1 -> obj_weight = 2.0 (Busca optimalidad)
        # Si max_viol es 100, exp(-50) = 0 -> obj_weight = 0.0 (Ignora el makespan, busca factibilidad)
        k_steepness = 0.5
        obj_weight = 2.0 * torch.exp(torch.tensor(-k_steepness * max_viol)).item()
            
        loss = (obj_weight * obj) + lagrange_term + penalty_term + (w_balance * load_std)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        current_real_viol = max_viol * 50.0

        if step % 50 == 0:
            training_history.append({
                'step': step,
                'makespan': obj.item(),
                'violation': current_real_viol
            })
        
        # Guardar el mejor modelo (Violacion 0 de KKTs)
        if current_real_viol < best_viol:
            best_viol = current_real_viol
            best_makespan = obj.item()
            best_model_state = copy.deepcopy(model.state_dict())
        elif current_real_viol == best_viol and obj.item() < best_makespan:
            best_makespan = obj.item()
            best_model_state = copy.deepcopy(model.state_dict())

        #Frecuencia dinámica de variables Duales (KKT) (MEJORA)
        #Actualiza cada 10 pasos al inicio, y CADA PASO después del step 5000
        freq_duals = 1 if step >= 5000 else 10
        if step % freq_duals == 0:
            with torch.no_grad():
                s_new, m_new = model(job_ids, tau=tau)
                g_new = build_constraints_v2(s_new, p_time_medical, p_time_occupancy, m_new)
                duals.update(g_new, rho)
                
        if step > 500 and step % rho_step == 0:
             rho = min(rho * beta, rho_max)
        
        if step % 1000 == 0:
            print(f"Step {step:04d} | Tau: {tau:.2f} | Obj: {obj.item():.0f} | Viol: {current_real_viol:.1f} | BalanceLoss: {load_std.item():.1f}")

    print("Restaurando pesos de la mejor solución encontrada...")
    model.load_state_dict(best_model_state)
    
    # Inferencia final
    model.eval()
    with torch.no_grad():
        s_pred_final, m_probs_final = model(job_ids, tau=0.001)
        
    return model, s_pred_final, m_probs_final, training_history