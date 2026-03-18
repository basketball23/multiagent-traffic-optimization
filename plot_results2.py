import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "evaluation-results/low-demand"
folders = {
    "Fixed-Time": os.path.join(base_dir, "baseline-fixed"),
    "Rule-Based": os.path.join(base_dir, "baseline-rule-based"),
    "Pedestrian-Aware MARL": os.path.join(base_dir, "marl")
}

def load_and_summarize(folder_path):
    """Reads CSVs, averages the entire simulation per seed, then averages across seeds."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"Warning: No CSV files found in {folder_path}")
        return None
    
    seed_averages = []
    for file in all_files:
        df = pd.read_csv(file)
        
        # Get the mean of all steps to represent this single seed's overall performance
        seed_mean = df.mean(numeric_only=True)
        seed_averages.append(seed_mean)
        
    # Combine all 10 seeds into one dataframe (10 rows)
    concat_seeds = pd.DataFrame(seed_averages)
    
    # Calculate the final mean and standard deviation across the 10 seeds for our separated metrics
    summary = {
        'veh_avg_mean': concat_seeds['vehicle_average_waiting_time'].mean(),
        'veh_avg_std': concat_seeds['vehicle_average_waiting_time'].std(),
        'ped_avg_mean': concat_seeds['pedestrian_average_waiting_time'].mean(),
        'ped_avg_std': concat_seeds['pedestrian_average_waiting_time'].std(),
        'cross_modal_mean': concat_seeds['cross_modal_fairness'].mean(),
        'cross_modal_std': concat_seeds['cross_modal_fairness'].std(),
        'intra_lane_mean': concat_seeds['intra_lane_fairness'].mean(),
        'intra_lane_std': concat_seeds['intra_lane_fairness'].std(),
        'p95_ped_mean': concat_seeds['p95_ped_wait_time'].mean(),
        'p95_ped_std': concat_seeds['p95_ped_wait_time'].std(),
    }
    return summary

print("Compiling summary statistics across all seeds...")

data = {}
for name, folder in folders.items():
    stats = load_and_summarize(folder)
    if stats is not None:
        data[name] = stats

plt.style.use('bmh') 
colors = {'Fixed-Time': '#2a9d8f', 'Rule-Based': '#457b9d', 'Pedestrian-Aware MARL': '#e63946'}
names = list(data.keys())

# Expanded to a 3x2 grid with increased height
fig, axs = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Traffic Control Model Performance (Simulation Averages Across 10 Seeds)', fontsize=18, fontweight='bold')

# Helper function to keep chart generation clean
def plot_bar_chart(ax, metric_prefix, title, ylabel, is_index=False):
    means = [data[name][f'{metric_prefix}_mean'] for name in names]
    stds = [data[name][f'{metric_prefix}_std'] for name in names]
    
    bars = ax.bar(names, means, yerr=stds, capsize=10, 
                  color=[colors[name] for name in names], alpha=0.8, edgecolor='black')
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    
    # Scale Y-axis to make room for text labels
    max_height = max([m + s for m, s in zip(means, stds)]) if means else 0
    if is_index:
        ax.set_ylim(0, 1.15) # Fixed bounds for 0.0 to 1.0 metrics
        max_height = 1.0
    elif max_height > 0:
        ax.set_ylim(0, max_height * 1.2)
        
    # Add data labels above the error bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        error_height = stds[i]
        
        if is_index:
            text_label = f"{yval:.2f}"
        else:
            text_label = f"{yval:.1f}s"
            
        # Offset text slightly above the error bar cap
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + error_height + (max_height * 0.03), 
                text_label, ha='center', va='bottom', fontweight='bold', color='black', fontsize=11)

# --- Graph 1: Vehicle Average Wait Time ---
plot_bar_chart(axs[0, 0], 'veh_avg', 'Vehicle Average Wait Time (Lower is Better)', 'Wait Time (Seconds)')

# --- Graph 2: Pedestrian Average Wait Time ---
plot_bar_chart(axs[0, 1], 'ped_avg', 'Pedestrian Average Wait Time (Lower is Better)', 'Wait Time (Seconds)')

# --- Graph 3: Cross-Modal Fairness ---
plot_bar_chart(axs[1, 0], 'cross_modal', 'Cross-Modal Fairness: Veh vs Ped Gap (Lower is Better)', 'Absolute Difference (s)')

# --- Graph 4: Intra-Lane Fairness ---
plot_bar_chart(axs[1, 1], 'intra_lane', "Intra-Lane Fairness: Jain's Index (Higher is Better)", 'Fairness Index (0.0 to 1.0)', is_index=True)

# --- Graph 5: P95 Pedestrian Wait Time ---
plot_bar_chart(axs[2, 0], 'p95_ped', 'Extreme Outliers: 95th Percentile Ped Wait (Lower is Better)', 'Wait Time (Seconds)')

# --- Hide the empty 6th subplot ---
fig.delaxes(axs[2, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('science_fair_results_summary.png', dpi=300)
print("Success! Saved summary bar charts to 'science_fair_results_summary.png'.")
plt.show()