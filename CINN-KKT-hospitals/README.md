# CINN-KKT Hospital Surgery Scheduler

# Objetivo de la rama
En esta rama se busca una optimizacion mayor del modelo buscando distintos objetivos (calidad de tiempo de espera, optimizacion de costos del hospital, makespan, balanceo de carga en hospital, etc.)

## Descripción del Proyecto

Este proyecto implementa un sistema de planificación automática de cirugías en hospitales utilizando:

- **CINN (Constraint-Informed Neural Networks)**: Redes neuronales que incorporan restricciones operacionales directamente en el modelo
- **Optimización Lagrangiana**: Método de variables duales para enforcar restricciones KKT
- **post procesamiento HC (Hill Climbing) o SA (Simulated Annealing)**: Post-procesamiento para refinar soluciones
- **Gumbel Softmax**: Relajación de decisiones discretas durante el entrenamiento

### Características Principales

- **Optimización Multi-Etapa**: Maneja 3 etapas quirúrgicas (preoperatorio, quirófano, postoperatorio)
- **Gestión de Recursos**: Asignación inteligente de pabellones con capacidad limitada (R=4)
- **Visualizaciones**: Gráficos Gantt detallados e histogramas de tiempos muertos
---

## Estructura del Proyecto

```
CINN-KKT-hospitals/
├── main.py                    # Punto de entrada principal
├── requirements.txt           # Dependencias del proyecto
├── README.md                  
├── LICENSE                    # Licencia del proyecto
│
├── src/                       # Módulos principales
│   ├── __init__.py
│   ├── data_loader.py         # Carga y preprocesamiento de datos
│   ├── model.py               # Arquitectura CINN (SchedulePINN)
│   ├── constraints.py         # Restricciones KKT y variables duales
│   ├── trainer.py             # Loop de entrenamiento con ADMM
│   ├── post_processing.py     # Decodificación y recocido simulado
│   ├── visualization.py       # Gráficos Gantt e histogramas
│   └── __pycache__/
│
└── data/
    ├── raw/
    │   └── 2_dataset_procesado_actualizado.csv
    └── processed/
        ├── solucion_final_optimizada.csv
        ├── gantt_final.png
        └── esperas_histograma.png
```

---

## Requisitos

- **Python 3.8+**
- **CUDA 11.0+** (recomendado para GPU, pero CPU también funciona)

---

## Instalación



### Crear entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
python -m pip install -r requirements.txt
```

---

## Uso

### Cargas
Cargar el .csv "2_dataset_procesado_actualizado.csv" en la carpeta "data/raw/" antes de correr el proyecto.

### Ejecución Completa

```bash
python main.py
```

Este comando ejecutará el pipeline completo:

1. **[1/5] Carga de Datos**: Lee y preprocesa el CSV hospitalario
2. **[2/5] Inicialización**: Crea la red neuronal CINN
3. **[3/5] Entrenamiento**: Entrena con restricciones KKT (10,000 steps)
4. **[4/5] Post-Procesamiento**: Aplica recocido simulado (5,000 iteraciones)
5. **[5/5] Generación de Reportes**: Guarda resultados y visualizaciones

### Parámetros Configurables

En `main.py` y `src/trainer.py` puedes ajustar:

```python
# data_loader.py
target_date = '2023-02-01'  # Fecha a analizar
num_samples = 16            # Número de pacientes
buffer_time = 20.0          # Buffer setup/cleanup (minutos)

# trainer.py
MAX_STEPS = 10000           # Iteraciones de entrenamiento
rho_start = 100.0           # Penalización inicial
tau_start = 2.0             # Temperatura Gumbel inicial
w_balance = 50.0            # Peso del balanceo de carga
```

---

## Salidas Generadas

El programa genera archivos en `data/processed/`:

### 1. **solucion_final_optimizada.csv**
Tabla con la solución optimizada:
- `job_id`: ID del paciente
- `stage_id`: Etapa (0=preoperatorio, 1=quirófano, 2=postoperatorio)
- `machine_id`: Pabellón asignado (1-4)
- `real_start`: Hora de inicio (minutos)
- `real_end`: Hora de fin (minutos)
- `dur_medical`: Duración de la intervención
- `dur_occupancy`: Ocupación total (intervención + buffers)

### 2. **gantt_final.png**
Gráfico Gantt visual con:
- Barras de color por paciente (intervención médica en gris)
- Buffers de setup (dorado) y cleanup (rojo)
- Líneas separadoras por etapa
- Makespan total en el título

### 3. **esperas_histograma.png**
Histogramas de distribución de tiempos muertos entre etapas

---

## Arquitectura del Modelo

### Red Neuronal CINN (SchedulePINN)

```
Entrada: ID del Paciente (embedding de 128 dims)
    ↓
[3 capas Tanh de 128 neuronas]
    ↓
Salida Dual:
  ├─ Start Times: softplus(W) * 300 → [J, I]
  └─ Machine Probs: Gumbel Softmax(logits) → [J, I, R]
```

### Función de Pérdida

$$\mathcal{L} = w_{obj} \cdot \text{makespan} + \lambda \cdot g^T \mu + \frac{\rho}{2} \|g_+\|^2 + w_{bal} \cdot \sigma(\text{cargas})$$

Donde:
- $g$: Violaciones de restricciones
- $\mu$: Variables duales (método ADMM)
- $\rho$: Penalización (crece con las iteraciones)
- $\sigma$: Desv. est. de carga de máquinas

### Restricciones Implementadas

1. **Secuencialidad**: $s_{j,i+1} \geq s_{j,i} + p_{j,i}$ ∀ j,i
2. **Tiempos de Espera**: $s_{j,i+1} - s_{j,i} - p_{j,i} \leq W_{max}$
3. **No Solapamiento**: Evita conflictos en recursos (probabilístico)

---

## Componentes Clave

### `data_loader.py`
- Limpieza de valores nulos
- Conversión de timestamps
- Cálculo de duraciones
- Filtrado por fecha específica

### `model.py`
- Definición de SchedulePINN
- Inicialización Xavier Normal
- Forward pass con relajación Gumbel Softmax

### `constraints.py`
- Construcción de restricciones KKT
- Variables duales (método ADMM)
- Cálculo de violaciones

### `trainer.py`
- Loop ADMM alternante
- Actualización de dual variables
- Annealing de temperatura τ y penalización ρ
- Guardado de mejor modelo

### `post_processing.py`
- Extracción de topología (decodificación de máquinas)
- Recocido simulado para refinamiento
- Intercambio de trabajos entre máquinas

### `visualization.py`
- Gráficos Gantt con buffers visualizados
- Histogramas de esperas por transición de etapa

---

## Referencias Técnicas

- **Informed Neural Networks**: Familia de Redes Neuronales informadas por distintas fuentes.
- **Constraint-Informed Neural Networks**: Incorporan restricciones matemáticas
- **Gumbel Softmax**: Relajación diferenciable de decisiones discretas (Maddison et al., 2017)
- **Job Shop Scheduling**: Problema NP-hard base del proyecto

---

## Licencia

Este proyecto está bajo la **Licencia MIT**.

```
MIT License

Copyright (c) 2026 PUCV Investigación

Se otorga permiso, sin costo, a cualquier persona que obtenga una copia
de este software y los archivos de documentación asociados (el "Software"),
para usar el Software sin restricción, incluyendo sin limitación los derechos
de usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o
vender copias del Software.

El Software se proporciona "TAL CUAL", sin garantía de ningún tipo.
```

---

## Contacto

nicolas.fuentes@pucv.cl
marcelo.becerra@pucv.cl
carlos.valle@pucv.cl

---

**Última actualización**: Febrero 2026