import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import warnings

def analyze_results(csv_path="raw_results.csv"):
    """
    Analyzes the benchmark results and generates a Q1-style statistical table.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run benchmark_runner.py first.")
        return

    # 1. Calculate BKS (Best Known Solution) per instance
    bks = df.groupby('Instance')['Makespan'].min().rename('BKS')
    df = df.merge(bks, on='Instance')

    # 2. Calculate RPD (Relative Percentage Deviation)
    # RPD = ((Makespan - BKS) / BKS) * 100
    df['RPD'] = ((df['Makespan'] - df['BKS']) / df['BKS']) * 100

    # 3. Descriptive Statistics
    # Agrupar por Instance y Algorithm
    summary = df.groupby(['Instance', 'Algorithm']).agg(
        RPD_Avg=('RPD', 'mean'),
        RPD_Best=('RPD', 'min'),
        RPD_Std=('RPD', 'std'),
        Time_Avg=('CPU_Time', 'mean')
    ).reset_index()

    # 4. Statistical Inference (Wilcoxon Test)
    # Comparar CINN (Control) vs cada Metaheurística
    instances = df['Instance'].unique()
    algorithms = [a for a in df['Algorithm'].unique() if a != 'CINN']
    
    # Preparamos un diccionario para guardar los símbolos de Wilcoxon
    # Clave: (Instance, Algorithm) -> Valor: (+), (-), (≈)
    wilcoxon_symbols = {}

    for instance in instances:
        cinn_rpd = df[(df['Instance'] == instance) & (df['Algorithm'] == 'CINN')]['RPD'].values
        
        for algo in algorithms:
            algo_rpd = df[(df['Instance'] == instance) & (df['Algorithm'] == algo)]['RPD'].values
            
            # Wilcoxon requires the same number of samples
            if len(cinn_rpd) != len(algo_rpd):
                wilcoxon_symbols[(instance, algo)] = "(?)"
                continue
            
            # Check if all differences are zero
            if np.all(cinn_rpd == algo_rpd):
                wilcoxon_symbols[(instance, algo)] = "(≈)"
                continue
            
            try:
                # alternative='two-sided' by default
                stat, p_val = wilcoxon(cinn_rpd, algo_rpd)
                
                if p_val < 0.05:
                    # Si p < 0.05, hay diferencia significativa
                    # Si CINN tiene menor media de RPD, CINN gana (+)
                    # Si CINN tiene mayor media de RPD, CINN pierde (-)
                    cinn_mean = np.mean(cinn_rpd)
                    algo_mean = np.mean(algo_rpd)
                    if cinn_mean < algo_mean:
                        wilcoxon_symbols[(instance, algo)] = "(+)"
                    else:
                        wilcoxon_symbols[(instance, algo)] = "(-)"
                else:
                    wilcoxon_symbols[(instance, algo)] = "(≈)"
            except Exception:
                # Manejar casos donde wilcoxon falla (ej. todas las diferencias iguales)
                wilcoxon_symbols[(instance, algo)] = "(≈)"

    # 5. Build Final Table
    # Queremos una fila por instancia y bloques de columnas por algoritmo
    # Formateamos RPD_Avg con el símbolo de Wilcoxon
    def format_rpd(row):
        if row['Algorithm'] == 'CINN':
            return f"{row['RPD_Avg']:.2f}"
        else:
            symbol = wilcoxon_symbols.get((row['Instance'], row['Algorithm']), "")
            return f"{row['RPD_Avg']:.2f} {symbol}"

    summary['RPD_Avg_Final'] = summary.apply(format_rpd, axis=1)

    # Pivotar para formato Q1
    # Filas: Instance
    # Columnas: Algorithm (RPD_Avg_Final, RPD_Best, Time_Avg)
    pivot_df = summary.pivot(index='Instance', columns='Algorithm', values=['RPD_Avg_Final', 'RPD_Best', 'Time_Avg'])

    # Reordenar columnas para que CINN esté primero
    cols = ['CINN'] + algorithms
    pivot_df = pivot_df.reindex(columns=cols, level=1)

    # 6. Print Markdown
    print("\n" + "="*80)
    print("      TABLA EXPERIMENTAL FINAL (Q1 JOURNAL FORMAT)")
    print("="*80)
    print(pivot_df.to_markdown())
    print("\nLeyenda:")
    print("(+): CINN es significativamente mejor (p < 0.05)")
    print("(-): CINN es significativamente peor (p < 0.05)")
    print("(≈): Sin diferencia significativa (p >= 0.05)")
    print("RPD: Relative Percentage Deviation (menor es mejor)")
    print("Time_Avg: Tiempo de CPU promedio en segundos")

if __name__ == "__main__":
    # Suprimir advertencias de Wilcoxon para diferencias cero
    warnings.filterwarnings("ignore", message="Sample size too small")
    warnings.filterwarnings("ignore", message="Exact p-value calculation is not possible")
    analyze_results()
