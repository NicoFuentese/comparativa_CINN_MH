import torch
import torch.nn as nn
import torch.nn.functional as F

#bloque residual
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # La magia de ResNet: Salida de la capa + Entrada original
        return torch.tanh(self.fc(x)) + x

#CINN de 3 capas y funcion de activacion tanh-ReLu
class SchedulePINN(nn.Module):
    def __init__(self, J, I, R, hidden_dim=128):
        super().__init__()
        self.J = J
        self.I = I
        self.R = R
        self.job_embedding = nn.Embedding(J, hidden_dim)
        
        #capa de entrada
        self.input_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        #Bloques Residuales en lugar de Sequential
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.res3 = ResidualBlock(hidden_dim)

        #cabezales
        self.head_start = nn.Linear(hidden_dim, I)
        self.head_machine = nn.Linear(hidden_dim, I * R)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, job_ids, tau=1.0):
        feats = self.job_embedding(job_ids)
        
        #flujo a traves de la red residual
        x = self.input_layer(feats)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Multiplicador 300 para abarcar tiempos largos reales (>20 horas)
        start_times = F.softplus(self.head_start(x)) * 300.0 
        
        machine_logits = self.head_machine(x).view(-1, self.I, self.R)
        
        if self.training:
            # Gumbel Softmax relaja la decisión discreta (necesario en CINN)
            machine_probs = F.gumbel_softmax(machine_logits, tau=tau, hard=False, dim=-1)
        else:
            machine_probs = F.softmax(machine_logits / 0.1, dim=-1)
        
        return start_times, machine_probs