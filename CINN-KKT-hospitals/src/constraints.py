import torch
import torch.nn.functional as F

class DualVariables:
    def __init__(self, num_constraints, device):
        self.mu = torch.zeros(num_constraints, device=device)
        self.device = device
        
    def update(self, g_val, rho):
        self.mu = torch.max(torch.zeros_like(self.mu), self.mu + rho * g_val)

def calculate_makespan(start_times, p_time):
    end_times = start_times + p_time
    return torch.max(end_times)


def build_constraints_v2(start_times, p_medical, p_occupancy, machine_probs):
    g_list = []
    
    # SECUENCIALIDAD (El inicio de etapa i+1 debe ser >= fin etapa i)
    for i in range(start_times.shape[1] - 1):
        g_seq = (start_times[:, i] + p_medical[:, i]) - start_times[:, i+1]
        g_list.append(g_seq)
        
    # TIEMPO MÁXIMO DE ESPERA (W_max = 60 minutos) penalizar si propone que un paciente espere más de 60 min entre etapas.
    W_max = 60.0 
    for i in range(start_times.shape[1] - 1):
        g_wait = (start_times[:, i+1] - (start_times[:, i] + p_medical[:, i])) - W_max
        g_list.append(g_wait)
        
    # OVERLAP (Solapamiento de Recursos incluyendo Setup/Cleanup - Bai 2022 Eq 4 y 7)
    penalty = 0.0
    J, I = start_times.shape

    # Bai (2022) estipula: Fin de Ocupación = x_J + p_J + cleanup_J + setup_J
    # Nuestro p_occupancy ya contiene esta suma (p_medical + buffer_time)
    end_times_occupancy = start_times + p_occupancy
    scale = 50.0 
    
    for i in range(I):
        # Inicio real de la ocupacion del recurso
        s = start_times[:, i].unsqueeze(1)
        #fin de la ocupacion (cleanup y setup incluido)
        e = end_times_occupancy[:, i].unsqueeze(1)
        
        inter_min_e = torch.min(e, e.T)
        inter_max_s = torch.max(s, s.T)
        
        # Calculo de solapamiento fisico en la maquina
        overlap_time = F.relu(inter_min_e - inter_max_s)
        
        mask = torch.eye(J, device=start_times.device).bool()
        overlap_time = overlap_time.masked_fill(mask, 0.0)
        
        # Superposición en el ESPACIO (misma maquina)
        m_probs_i = machine_probs[:, i, :]
        resource_conflict = torch.matmul(m_probs_i, m_probs_i.T)
        
        penalty += (overlap_time * resource_conflict).sum()
    
    g_overlap = penalty.unsqueeze(0) / scale
    
    # Unir todas las restricciones lineales
    g_vec = torch.cat(g_list, dim=0)
    
    # Retornar vector maestro de violaciones KKT
    return torch.cat([g_vec, g_overlap], dim=0)