import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "evaluation-results/low-demand"
folders = {
    "Fixed-Time": os.path.join(base_dir, "baseline-fixed"),
    "Rule-Based": os.path.join(base_dir, "baseline-rule-based"),
    "Pedestrian-Aware MARL": os.path.join(base_dir, "marl")
}

def parse_summary_file(filepath):
    """Extracts metrics from a single text file using regular expressions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Regex patterns look for the text label, any amount of whitespace (\s*), 
    # the number ([\d\.]+), and optionally the 's' unit
    try:
        return {
            'veh_avg': float(re.search(r"True Avg Veh Wait Time:\s*([\d\.]+)s?", content).group(1)),
            'ped_avg': float(re.search(r"True Avg Ped Wait Time:\s*([\d\.]+)s?", content).group(1)),
            'cross_modal': float(re.search(r"FINAL CROSS-MODAL GAP:\s*([\d\.]+)s?", content).group(1)),
            'intra_lane': float(re.search(r"TRUE INTRA-LANE FAIRNESS:\s*([\d\.]+)", content).group(1)),
            'p95_ped': float(re.search(r"True 95th Percentile Ped:\s*([\d\.]+)s?", content).group(1))
        }
    except (AttributeError, ValueError) as e:
        print(f"Error parsing metrics in file {filepath}. Make sure the formatting matches exactly.")
        return None

def load_and_summarize(folder_path):
    """Reads TXT files, extracts metrics per seed, then averages across all seeds."""
    # Updated to look for text files
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not all_files:
        print(f"Warning: No TXT files found in {folder_path}")
        return None
    
    seed_records = []
    for file in all_files:
        metrics = parse_summary_file(file)
        if metrics is not None:
            seed_records.append(metrics)
            
    if not seed_records:
        return None
        
    # Combine all 10 seeds into one dataframe (10 rows, 5 columns)
    concat_seeds = pd.DataFrame(seed_records)
    
    # Calculate the final mean and standard deviation across the seeds
    summary = {
        'veh_avg_mean': concat_seeds['veh_avg'].mean(),
        'veh_avg_std': concat_seeds['veh_avg'].std(),
        'ped_avg_mean': concat_seeds['ped_avg'].mean(),
        'ped_avg_std': concat_seeds['ped_avg'].std(),
        'cross_modal_mean': concat_seeds['cross_modal'].mean(),
        'cross_modal_std': concat_seeds['cross_modal'].std(),
        'intra_lane_mean': concat_seeds['intra_lane'].mean(),
        'intra_lane_std': concat_seeds['intra_lane'].std(),
        'p95_ped_mean': concat_seeds['p95_ped'].mean(),
        'p95_ped_std': concat_seeds['p95_ped'].std(),
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
fig.suptitle('Traffic Control Model Performance (Simulation Averages Across Seeds)', fontsize=18, fontweight='bold')

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