# MH-Hospitals: Metaheuristicas para Programacion de Cirugias

## Descripcion

Framework de metaheuristicas para el problema de **Three-Station Job Shop Scheduling (TSJS)** aplicado a la programacion de cirugias hospitalarias. El sistema modela tres etapas quirurgicas con asignacion dinamica de personal:

1. **APR** (Pre-operatorio) -- Anestesiologos
2. **OR** (Quirofano) -- Cirujanos
3. **ARR** (Recuperacion post-operatoria) -- Medicos de recuperacion

### Algoritmos Implementados

| Algoritmo | Tipo | Descripcion |
|-----------|------|-------------|
| **GA** | Genetic Algorithm | Basado en poblacion con crossover, mutacion y elitismo |
| **dPSO** | Discrete Particle Swarm Optimization | PSO adaptado a espacio discreto con funcion sigmoide |
| **SBOA** | Secretary Bird Optimization Algorithm | Inspirado en naturaleza, con Levy flights |
| **dMShOA** | Discrete Mantis Shrimp Optimization Algorithm | Basado en sigmoide para decision discreta |

### Modos de Simulacion

- **Modo electivo**: Solo cirugias programadas
- **Modo emergencia** (TSJS): Cirugias electivas + emergencias que irrumpen, con scheduler dinamico

---

## Estructura del Proyecto

```
MH-hospitals/
├── main.py                     # Punto de entrada
├── requirements.txt            # Dependencias
├── check_config.py             # Validacion de configuracion
│
├── algorithms/                 # Metaheuristicas
│   ├── ga.py                   # Genetic Algorithm
│   ├── dpso.py                 # Discrete PSO
│   ├── sboa.py                 # Secretary Bird Optimization
│   └── dmshoa.py               # Discrete Mantis Shrimp Optimization
│
├── config/                     # Configuracion
│   ├── config.json             # Parametros centralizados
│   ├── config.py               # Cargador de configuracion
│   ├── config.quick.json       # Configuracion rapida (pruebas)
│   └── algorithms_loader.py    # Carga dinamica de algoritmos
│
├── simulation/                 # Motor de simulacion
│   ├── scheduler.py            # Construccion de horarios y fitness
│   ├── dynamic_scheduler.py    # Scheduler dinamico para emergencias
│   └── emergency_generator.py  # Generador de cirugias de emergencia
│
├── core/                       # Orquestacion
│   ├── simulation_runner.py    # Ejecucion paralela (joblib)
│   ├── report_generator.py     # Reportes estadisticos
│   └── file_manager.py         # Gestion de directorios de salida
│
├── workers/                    # Workers por simulacion
│   ├── elective_worker.py
│   └── emergency_worker.py
│
├── data/                       # Datos
│   ├── data_loader.py          # Carga de datos reales
│   ├── data_generator.py       # Generacion de datos sinteticos
│   └── real_surgeries_data.csv # Dataset real (>10 MB, excluido del repo)
│
├── utils/                      # Utilidades
│   ├── statistics.py           # Analisis estadistico
│   ├── reporting.py            # Generacion de reportes
│   ├── plotting.py             # Visualizaciones
│   └── logger.py               # Logging
│
├── results/                    # Resultados generados
│   ├── elective/
│   │   ├── csv/                # Resultados por algoritmo
│   │   └── plots/              # Gantt, convergencia, histogramas, boxplots
│   └── emergencies/
│       ├── csv/
│       └── plots/
│
└── legacy/                     # Codigo heredado
    └── main_old.py
```

---

## Requisitos

- Python 3.8+
- Dependencias: NumPy, SciPy, Pandas, Matplotlib, Joblib

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
python -m pip install -r requirements.txt
```

---

## Uso

### Configuracion

Todos los parametros se ajustan en `config/config.json`:

- `experiment.num_simulations`: Numero de ejecuciones Monte Carlo (default: 50)
- `experiment.num_surgeries`: Cantidad de cirugias por simulacion (default: 8)
- `resources`: Salas por etapa (default: 4 APR, 4 OR, 4 ARR)
- `algorithms.*.enabled`: Activar/desactivar algoritmos individualmente
- `emergencies.enabled`: Activar modo emergencia (default: false)

### Ejecucion

```bash
# Modo electivo (sin emergencias)
python main.py

# Modo emergencia (cambiar emergencies.enabled a true en config.json)
python main.py
```

### Salidas Generadas

En `results/elective/` (o `results/emergencies/`):

- **CSVs**: Mejores horarios, estrategias, analisis estadistico, resumen
- **Graficos Gantt**: Visualizacion de horarios optimos por algoritmo
- **Convergencia**: Curvas de convergencia por simulacion
- **Boxplots**: Comparacion de makespan entre algoritmos
- **Histogramas**: Distribucion de tiempos de espera
- **Barras**: Tiempos de ejecucion por algoritmo

---

## Funcion Objetivo

$$\mathcal{L} = \alpha \cdot \sum s_{j,1} + \beta \cdot \sum w_j + \gamma \cdot \max(w_j) + \delta \cdot \sigma(\text{cargas})$$

Donde:
- $s_{j,1}$: Tiempo de inicio de la cirugia $j$
- $w_j$: Tiempo de espera entre etapas
- $\sigma(\text{cargas})$: Desbalance de carga entre salas
- $\alpha, \beta, \gamma, \delta$: Pesos configurables
