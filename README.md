# Pipeline de Benchmarking: CINN vs. Metaheurísticas

Este repositorio contiene el framework de evaluación comparativa entre el modelo **CINN** (Constrained-Informed Neural Network) y cuatro metaheurísticas estándar (**GA, dPSO, SBOA, dMShOA**) para el problema de programación de quirófanos (Three-Station Job Shop Scheduling).

## 🚀 Guía de Uso Rápido

### 1. Requisitos e Instalación
El sistema utiliza dos entornos virtuales existentes para gestionar las dependencias de cada proyecto (especialmente Torch para CINN y Scipy para las estadísticas).

```bash
# Instalar tabulate para reportes en Markdown (en el entorno de MH)
MH-hospitals/.venv/Scripts/python -m pip install tabulate -q
```

### 2. Ejecución del Experimento
Para generar los datos crudos (30 ejecuciones por algoritmo con 16 cirugías reales):

```bash
CINN-KKT-hospitals/.venv/Scripts/python benchmark_runner.py
```
*Esto generará el archivo `raw_results.csv`.*

### 3. Análisis Estadístico
Para generar la tabla de resultados finales con el Test de Wilcoxon:

```bash
MH-hospitals/.venv/Scripts/python metrics_analyzer.py
```

---

## 📊 Interpretación de Métricas

Para validar científicamente los resultados, utilizamos las siguientes métricas:

1.  **BKS (Best Known Solution):** Es el tiempo mínimo (Makespan) encontrado entre todos los algoritmos durante todas las pruebas. Es nuestro "punto de perfección".
2.  **RPD (Relative Percentage Deviation):** Indica qué tan lejos (%) está un algoritmo del BKS.
    *   *Fórmula:* `((Makespan - BKS) / BKS) * 100`
    *   **0.00%** es la solución perfecta. Valores altos indican ineficiencia.
3.  **RPD_Avg:** Consistencia del algoritmo a lo largo de las 30 ejecuciones.
4.  **RPD_Best:** El mejor resultado individual logrado. Indica si el algoritmo es capaz de encontrar el óptimo global.
5.  **Test de Wilcoxon (+, -, ≈):** Validación estadística con **p-value < 0.05**.
    *   **(+)**: CINN es significativamente mejor (victoria matemática, no por azar).
    *   **(-)**: CINN es significativamente peor.
    *   **(≈)**: Empate estadístico.

---

## 📈 Resultados Obtenidos (Instancia Real: 01-02-2023)

| Algoritmo | RPD_Avg | RPD_Best | Time_Avg (s) | Inferencia |
| :--- | :--- | :--- | :--- | :--- |
| **CINN (Propuesto)** | **5.49%** | **0.00%** | 81.94 | - |
| **GA** | 59.02% | 40.44% | 20.83 | **CINN (+)** |
| **dPSO** | 74.62% | 35.86% | 21.62 | **CINN (+)** |
| **SBOA** | 64.08% | 39.58% | 38.60 | **CINN (+)** |
| **dMShOA** | 59.54% | 43.05% | 22.37 | **CINN (+)** |

### Conclusiones Técnicas para el Artículo:
*   **Precisión:** El modelo CINN demostró una precisión superior, manteniéndose en promedio a solo un **5.49%** del óptimo, mientras que las metaheurísticas fallaron con desviaciones superiores al **50%**.
*   **Capacidad de Optimización:** CINN fue el único que logró el **RPD 0.00%**, encontrando la mejor configuración de quirófanos posible para ese día.
*   **Significancia:** El símbolo **(+)** en todos los comparativos confirma que la superioridad de CINN es estadísticamente significativa para una publicación de alto impacto.

---

## 🛠 Estructura del Pipeline
*   `benchmark_runner.py`: Orquestador que carga los datos reales, entrena la red CINN y ejecuta las metaheurísticas bajo las mismas condiciones de semillas y datos.
*   `metrics_analyzer.py`: Procesa `raw_results.csv` para realizar los tests no paramétricos y formatear la tabla Markdown.
*   `raw_results.csv`: Base de datos de experimentos (150 registros).

---

## Estructura del Repositorio

### CINN-KKT-hospitals/

Modelo de optimizacion basado en **Constraint-Informed Neural Networks (CINN)** con condiciones KKT y ADMM. Resuelve el problema de Three-Station Job Shop Scheduling para programacion de cirugias hospitalarias.

- **Tecnologias**: PyTorch, Pandas, NumPy, Matplotlib, Seaborn
- **Version de referencia**: PyTorch 2.10.0+cpu (los resultados pueden variar segun la version de PyTorch)
- **Post-procesamiento**: Hill Climbing / Simulated Annealing

> Documentacion detallada en [CINN-KKT-hospitals/README.md](CINN-KKT-hospitals/README.md)

### MH-hospitals/

Framework de metaheuristicas para el mismo problema de scheduling quirurgico, con simulacion de cirugias electivas y de emergencia.

- **Algoritmos**: GA (Genetic Algorithm), dPSO (Discrete Particle Swarm Optimization), SBOA, dMShOA
- **Tecnologias**: NumPy, SciPy, Pandas, Matplotlib, Joblib
- Incluye generacion de reportes, visualizaciones y analisis estadistico.
