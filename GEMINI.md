# 🤖 Prompt para Agente de Codificación: Pipeline de Benchmarking y Estadística Q1

**Rol del Agente:** Eres un Senior Software Engineer especializado en Machine Learning, Investigación Operativa (Operations Research) y Análisis Estadístico.

**Contexto:** Estamos redactando un artículo científico (nivel Q1) comparando un modelo CINN (Constrained-Informed Neural Network) contra 4 metaheurísticas estándar (ej. GA, TS, SA, ACO) para el problema de *Three-Station Job Shop Scheduling*. Necesitamos construir un pipeline completo en Python que genere los datos experimentales y construya la matriz estadística final.

**Instrucción Principal:** Tu objetivo es escribir dos scripts modulares (`benchmark_runner.py` y `metrics_analyzer.py`). Tienes libertad para tomar decisiones técnicas u optimizar el código si detectas fallos lógicos en este plan (ej. manejo de empates en Wilcoxon o problemas de precisión en Pandas).

---

### Fase 1: Generación de Datos (`benchmark_runner.py`)

**Objetivo:** Crear un framework que ejecute los algoritmos y genere un dataset estandarizado.

1. **Wrappers/Mocks:** Crea interfaces o funciones *mock* para la CINN y las 4 metaheurísticas. Deben recibir `(instancia, semilla)` y retornar el `Makespan` (asumiendo 100% factibilidad vía decodificador). Deja claro dónde debo inyectar mis modelos reales.
2. **Parámetros del Experimento:** - `NUM_RUNS = 30` (Mínimo estricto para validez estadística).
   - `INSTANCES = ['small_8x3', 'medium_15x3', 'medium_16x3']`.
3. **Ejecución:** Implementa bucles para iterar sobre Instancias -> Algoritmos -> Runs.
   - Usa semillas reproducibles (`seed = run_index`).
   - Mide el `CPU_Time` usando `time.perf_counter()`, aislando estrictamente el tiempo de cómputo del tiempo de lectura de I/O.
4. **Entregable de esta fase:** Exportar un archivo `raw_results.csv` con las columnas exactas: `['Instance', 'Algorithm', 'Run', 'Makespan', 'CPU_Time']`.

---

### Fase 2: Análisis Estadístico y Tabla Q1 (`metrics_analyzer.py`)

**Objetivo:** Procesar el `raw_results.csv` para generar la tabla final en formato Markdown con inferencia estadística integrada.

1. **Métricas Base (Estadística Descriptiva):**
   - Calcula el **BKS** (Best Known Solution / Mínimo Makespan global) para cada instancia.
   - Transforma los valores de Makespan a **RPD** (Relative Percentage Deviation): `((Makespan - BKS) / BKS) * 100`.
   - Agrupa por `Instance` y `Algorithm` calculando: `RPD_Avg`, `RPD_Best`, `Std` (sobre RPD) y `Time_Avg`.
2. **Inferencia Estadística (Tests No Paramétricos):**
   - *Opcional pero recomendado:* Aplica el Test de Friedman por instancia para validar diferencias globales.
   - *Obligatorio:* Aplica el **Test de Wilcoxon Pareado** (`scipy.stats.wilcoxon`) usando la CINN como "Algoritmo de Control" frente a cada metaheurística (CINN vs MH1, CINN vs MH2, etc.).
3. **Asignación de Simbología:**
   - Si `p-value < 0.05` y CINN gana (menor RPD) -> **(+)**
   - Si `p-value < 0.05` y CINN pierde -> **(-)**
   - Si `p-value >= 0.05` -> **(≈)**
4. **Entregable de esta fase:** Un script que imprima en consola una tabla **Markdown** limpia (usando Pandas `to_markdown` o similar). La columna de `RPD_Avg` de las metaheurísticas debe concatenar el valor numérico con el símbolo (ej. `2.45 (+)`). La estructura debe mostrar las instancias en las filas y los bloques de algoritmos (con sus métricas) en las columnas.

---
**Nota para el agente:** Asegúrate de documentar el código y manejar advertencias comunes de `scipy` (ej. cuando todas las diferencias en Wilcoxon son cero). ¡Manos a la obra!