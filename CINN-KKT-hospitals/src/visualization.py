import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os
import seaborn as sns

# Configurar el estilo científico de Seaborn
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

#ploteo de carta gantt
def plot_advanced_gantt(df, makespan, J, setup_mins=10.0, output_path="data/processed/gantt_final.png"):
    fig, ax = plt.subplots(figsize=(20, 12))
    colors = plt.cm.get_cmap('tab20', J)
    
    for _, row in df.iterrows():
        start = row['real_start']
        dur_med = row['dur_medical']
        dur_occ = row['dur_occupancy']
        y = row['global_machine_id']
        j = int(row['job_id'])
        
        buffer_total = dur_occ - dur_med
        setup = setup_mins if buffer_total >= setup_mins else buffer_total / 2.0
        cleanup = buffer_total - setup
        
        # Setup
        ax.add_patch(patches.Rectangle((start, y - 0.4), setup, 0.8, 
                                       linewidth=0.5, edgecolor='black', facecolor='#FFD700', alpha=0.8))
        # Cirugía
        ax.add_patch(patches.Rectangle((start + setup, y - 0.4), dur_med, 0.8, 
                                     linewidth=1, edgecolor='black', facecolor=colors(j), alpha=0.9))
        # Cleanup
        ax.add_patch(patches.Rectangle((start + setup + dur_med, y - 0.4), cleanup, 0.8, 
                                       linewidth=0.5, edgecolor='black', facecolor='#FF6347', alpha=0.8))
        
        if dur_med > 30:
            ax.text(start + setup + dur_med/2, y, f"P{j}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # Legend
    setup_patch = patches.Patch(color='#FFD700', alpha=0.8, label='Setup (Room Preparation)')
    med_patch = patches.Patch(color='gray', alpha=0.9, label='Surgery / Intervention')
    clean_patch = patches.Patch(color='#FF6347', alpha=0.8, label='Cleanup (Cleaning)')
    ax.legend(handles=[setup_patch, med_patch, clean_patch], loc='upper right', fontsize=12)

    ax.set_yticks(range(1, 13))
    labels = []
    for r in range(1, 13):
        if r <= 4: labels.append(f"PRE-{r}")
        elif r <= 8: labels.append(f"QX-{r-4}")
        else: labels.append(f"POST-{r-8}")
        
    ax.set_yticklabels(labels, fontweight='bold')
    ax.axhline(y=4.5, color='black', linestyle='-', linewidth=2)
    ax.axhline(y=8.5, color='black', linestyle='-', linewidth=2)
    
    ax.set_xlabel("Time (Minutes)", fontsize=14, fontweight='bold')
    ax.set_title(f"Actual Surgical Planning (CINN + SA) - Makespan: {makespan:.0f} min\nVisual Detail: Setup (10m) + Surgery + Cleanup", fontsize=16)
    
    ax.set_xlim(-makespan*0.02, makespan * 1.02)
    ax.set_ylim(0.5, 12.5)
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gantt chart saved to: {output_path}")
    plt.close()

#ploteo KPI de histogramas
def plot_wait_histograms(df, output_path="data/processed/esperas_histograma.png"):
    waits_dict = {}
    stages = sorted(df['stage_id'].unique())

    for i in range(len(stages) - 1):
        stage_curr, stage_next = stages[i], stages[i+1]
        delays = []
        for j in df['job_id'].unique():
            row_curr = df[(df['job_id'] == j) & (df['stage_id'] == stage_curr)]
            row_next = df[(df['job_id'] == j) & (df['stage_id'] == stage_next)]
            if row_curr.empty or row_next.empty: continue
            
            end_curr = row_curr['real_end'].values[0]
            start_next = row_next['real_start'].values[0]
            delays.append(start_next - end_curr)
            
        waits_dict[(stage_curr, stage_next)] = delays

    colors = ['#2ca02c', '#1f77b4']
    stage_names = {0: "Preoperative", 1: "Operating Room", 2: "Postoperative"}

    fig, axes = plt.subplots(1, len(waits_dict), figsize=(12, 5), sharey=True)
    if len(waits_dict) == 1: axes = [axes]

    for idx, ((i, ip1), lst) in enumerate(waits_dict.items()):
        ax = axes[idx]
        arr = np.array(lst, dtype=float)
        ax.hist(arr, bins=10, alpha=0.75, color=colors[idx % len(colors)], edgecolor='black')
        
        mean_val = np.mean(arr)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.1f} min")

        ax.set_title(f"Wait Times: {stage_names.get(i)} $\\to$ {stage_names.get(ip1)}")
        ax.set_xlabel("Wait Time (min)")
        if idx == 0: ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.suptitle("Distribution of Patient Idle Times", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Histograms saved to: {output_path}")
    plt.close()

def generar_estadisticas_bai(csv_path="data/processed/solucion_final_optimizada.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run the model first.")
        return

    makespan = df['real_end'].max()
    
    # ---------------------------------------------------------
    # CÁLCULO DE TIEMPOS DE ESPERA (Wait Times)
    # ---------------------------------------------------------
    waits = []
    for j in df['job_id'].unique():
        paciente_df = df[df['job_id'] == j].sort_values('stage_id')
        
        end_pre = paciente_df[paciente_df['stage_id'] == 0]['real_end'].values
        start_qx = paciente_df[paciente_df['stage_id'] == 1]['real_start'].values
        
        end_qx = paciente_df[paciente_df['stage_id'] == 1]['real_end'].values
        start_post = paciente_df[paciente_df['stage_id'] == 2]['real_start'].values
        
        if len(end_pre)>0 and len(start_qx)>0:
            waits.append({'job_id': j, 'Phase': 'PRE -> QX', 'Wait (min)': start_qx[0] - end_pre[0]})

        if len(end_qx)>0 and len(start_post)>0:
            waits.append({'job_id': j, 'Phase': 'QX -> POST', 'Wait (min)': start_post[0] - end_qx[0]})
            
    df_waits = pd.DataFrame(waits)

    # ---------------------------------------------------------
    # CÁLCULO DE UTILIZACIÓN DE RECURSOS (Load Balancing)
    # ---------------------------------------------------------
    # Utilización = (Tiempo total de ocupación médica en el pabellón) / Makespan
    utilization = []
    for m in df['resource_name'].unique():
        df_maq = df[df['resource_name'] == m]
        # Sumamos la duración médica (el trabajo real)
        tiempo_trabajo = df_maq['dur_medical'].sum()
        utilizacion_pct = (tiempo_trabajo / makespan) * 100
        etapa = m.split('-')[0] # Extrae "PRE", "QX" o "POST"
        utilization.append({'Resource': m, 'Stage': etapa, 'Utilization (%)': utilizacion_pct})
        
    df_util = pd.DataFrame(utilization)

    # =========================================================
    # FIGURA 1: Histograma y KDE de Esperas (Bai Style)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_waits, x='Wait (min)', hue='Phase', kde=True,
                 palette=['#1f77b4', '#ff7f0e'], bins=15, alpha=0.6, ax=ax1)
    
    ax1.set_title("Distribution of Wait Times Between Surgical Stages", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Wait Time (Minutes)", fontweight='bold')
    ax1.set_ylabel("Frequency (N° Patients)", fontweight='bold')
    
    # --- CORRECCIÓN AQUÍ: Uso de PRE, QX y POST ---
    mean_pre_qx = df_waits[df_waits['Phase'] == 'PRE -> QX']['Wait (min)'].mean()
    mean_qx_post = df_waits[df_waits['Phase'] == 'QX -> POST']['Wait (min)'].mean()
    ax1.axvline(mean_pre_qx, color='#1f77b4', linestyle='--', label=f'Mean PRE->QX: {mean_pre_qx:.1f}m')
    ax1.axvline(mean_qx_post, color='#ff7f0e', linestyle='--', label=f'Mean QX->POST: {mean_qx_post:.1f}m')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig("data/processed/fig1_hist_esperas.png", dpi=300)
    plt.show()

    # =========================================================
    # FIGURA 2: Boxplot de Balanceo de Carga (Load Balancing)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Boxplot para mostrar la dispersión por etapa
# Cambia a esto:
    sns.boxplot(data=df_util, x='Stage', y='Utilization (%)', hue='Stage', legend=False, palette="Set2", width=0.5, ax=ax2, boxprops=dict(alpha=0.8))    # Stripplot para mostrar los puntos individuales (cada pabellón)
    sns.stripplot(data=df_util, x='Stage', y='Utilization (%)', color='black', alpha=0.7, jitter=True, size=8, ax=ax2)
    
    ax2.set_title("Load Balancing per Stage", fontweight='bold', fontsize=14)
    ax2.set_xlabel("Clinical Phase", fontweight='bold')
    ax2.set_ylabel("Resource Utilization (%) over Makespan", fontweight='bold')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig("data/processed/fig2_boxplot_utilizacion.png", dpi=300)
    #plt.show()

    # =========================================================
    # FIGURA 3: Boxplot del Flow Time del Paciente
    # =========================================================
    flow_times = []
    for j in df['job_id'].unique():
        paciente_df = df[df['job_id'] == j]
        inicio_absoluto = paciente_df['real_start'].min()
        fin_absoluto = paciente_df['real_end'].max()
        tiempo_ideal = paciente_df['dur_medical'].sum()
        
        flow_times.append({
            'Patient': f"P{j}",
            'Real Hospital Time (min)': fin_absoluto - inicio_absoluto,
            'Ideal Surgical Time (min)': tiempo_ideal
        })
        
    df_flow = pd.DataFrame(flow_times)
    df_melted = df_flow.melt(id_vars=['Patient'], var_name='Metric', value_name='Minutes')

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    # Cambia a esto:
    sns.boxplot(data=df_melted, x='Metric', y='Minutes', hue='Metric', legend=False, palette="Pastel1", width=0.5, ax=ax3)
    sns.stripplot(data=df_melted, x='Metric', y='Minutes', color='black', alpha=0.6, jitter=True, ax=ax3)

    ax3.set_title("Patient Efficiency: Real vs Ideal Time", fontweight='bold', fontsize=14)
    ax3.set_xlabel("", fontweight='bold')
    ax3.set_ylabel("Minutes", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("data/processed/fig3_boxplot_flow_time.png", dpi=300)
    plt.show()

    # --- CORRECCIÓN AQUÍ: Uso de PRE, QX y POST ---
    print("================ STATISTICAL SUMMARY ================")
    print(f"Total Makespan: {makespan:.2f} min")
    print(f"Average Wait PRE -> QX: {mean_pre_qx:.2f} min")
    print(f"Average Wait QX -> POST: {mean_qx_post:.2f} min")
    print(f"Average Utilization PRE: {df_util[df_util['Stage']=='PRE']['Utilization (%)'].mean():.1f}%")
    print(f"Average Utilization QX: {df_util[df_util['Stage']=='QX']['Utilization (%)'].mean():.1f}%")
    print(f"Average Utilization POST: {df_util[df_util['Stage']=='POST']['Utilization (%)'].mean():.1f}%")
    print("===================================================================")

def plot_convergence_curve(history_data, output_path="data/processed/convergencia_primal_dual.png"):
    # Convertimos la lista de diccionarios en un DataFrame
    df = pd.DataFrame(history_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # EJE Y IZQUIERDO: Makespan (Azul)
    color1 = '#1f77b4' # Azul científico
    ax1.set_xlabel('Training Iterations (Steps)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Predicted Makespan (Minutes)', color=color1, fontweight='bold', fontsize=12)
    line1 = ax1.plot(df['step'], df['makespan'], color=color1, label='Makespan (Efficiency)', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

    # EJE Y DERECHO: Violación KKT (Rojo)
    ax2 = ax1.twinx()  # Instanciar un segundo eje que comparte el mismo eje X
    color2 = '#d62728' # Rojo científico
    ax2.set_ylabel('Maximum KKT Constraint Violation', color=color2, fontweight='bold', fontsize=12)
    line2 = ax2.plot(df['step'], df['violation'], color=color2, label='Violations (Feasibility)', linewidth=2.5, alpha=0.85)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Escala Logarítmica (Simétrica) para el eje de violaciones porque los números son muy grandes
    ax2.set_yscale('symlog') 

    # Añadir leyenda combinada abajo
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12)

    plt.title('Primal-Dual Convergence Curve of CINN Network', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence curve saved to: {output_path}")
    plt.close()