import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = "evaluation_results/regular-demand-1x"
folders = {
    "Fixed-Time": os.path.join(base_dir, "baseline_fixed"),
    "Rule-Based": os.path.join(base_dir, "baseline_fixed"),
    "Pedestrian-Aware MARL": os.path.join(base_dir, "marl2")
}

def load_and_aggregate(folder_path):
    """reads all CSVs in a folder, averages them across the 10 seeds."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"Warning: No CSV files found in {folder_path}")
        return None
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
        
    concat_df = pd.concat(df_list)
    
    grouped = concat_df.groupby('step').agg(['mean', 'std'])
    
    grouped.columns = ['_'.join(col) for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    return grouped

print("Compiling data across all seeds...")

data = {}
for name, folder in folders.items():
    df = load_and_aggregate(folder)
    if df is not None:
        data[name] = df

plt.style.use('bmh') 
colors = {'Fixed-Time': '#2a9d8f', 'Rule-Based': '#457b9d', 'Pedestrian-Aware MARL': '#e63946'}

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Traffic Control Model Performance (1-Hour Simulation Averaged Across 10 Seeds)', fontsize=18, fontweight='bold')

# --- Graph 1: Overall Efficiency (Average Wait Time) ---
ax = axs[0, 0]
for name, df in data.items():
    overall_wait_mean = (df['vehicle_average_waiting_time_mean'] + df['pedestrian_average_waiting_time_mean']) / 2
    ax.plot(df['step'], overall_wait_mean, label=name, color=colors[name], linewidth=2)

ax.set_title('Overall Efficiency (Lower is Better)')
ax.set_xlabel('Simulation Step (Seconds)')
ax.set_ylabel('Average Vehicle Wait Time (Seconds)')
ax.legend()

# --- Graph 2: Cross-Modal Fairness ---
ax = axs[0, 1]
for name, df in data.items():
    ax.plot(df['step'], df['cross_modal_fairness_mean'], label=name, color=colors[name], linewidth=2)
    ax.fill_between(df['step'], 
                    df['cross_modal_fairness_mean'] - df['cross_modal_fairness_std'],
                    df['cross_modal_fairness_mean'] + df['cross_modal_fairness_std'], 
                    color=colors[name], alpha=0.1)

ax.set_title('Cross-Modal Fairness: Veh vs Ped Wait Gap (Lower is Better)')
ax.set_xlabel('Simulation Step (Seconds)')
ax.set_ylabel('Absolute Difference in Average Wait Times (s)')
ax.legend()

# --- Graph 3: Intra-Lane Fairness (Jain's Index) ---
ax = axs[1, 0]
for name, df in data.items():
    ax.plot(df['step'], df['intra_lane_fairness_mean'], label=name, color=colors[name], linewidth=2)

ax.set_title("Intra-Lane Fairness: Jain's Index (Higher is Better)")
ax.set_xlabel('Simulation Step (Seconds)')
ax.set_ylabel('Fairness Index (0.0 to 1.0)')
ax.set_ylim(0, 1.1)
ax.legend()

# --- Graph 4: P95 Pedestrian Wait Time (Bar Chart Summary) ---
ax = axs[1, 1]
names = list(data.keys())
p95_means = [data[name]['p95_ped_wait_time_mean'].mean() for name in names]
p95_stds = [data[name]['p95_ped_wait_time_mean'].std() for name in names] # Std dev of the means

bars = ax.bar(names, p95_means, yerr=p95_stds, capsize=10, 
              color=[colors[name] for name in names], alpha=0.8, edgecolor='black')

ax.set_title('Extreme Outliers: Average 95th Percentile Pedestrian Wait')
ax.set_ylabel('Pedestrian Wait Time (Seconds)')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{round(yval, 1)}s", ha='center', va='bottom', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('science_fair_results.png', dpi=300)
print("Success! Saved high-resolution graph to 'science_fair_results.png'.")
plt.show()