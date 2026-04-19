import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "evaluation-results-nema"
demands = ["low-demand", "regular-demand", "high-demand"]

models = {
    "Fixed-Time": "baseline-fixed",
    "Ablation": "marl-no-rew",
    "Proposed PPO": "marl"
}

colors = {
    'Fixed-Time': '#9ca3af',
    'Ablation': '#f87171',
    'Proposed PPO': '#10b981'
}

# --- 2. Parsing Functions ---
def parse_summary_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        return {
            'veh_avg': float(re.search(r"True Avg Veh Wait Time:\s*([\d\.]+)s?", content).group(1)),
            'ped_avg': float(re.search(r"True Avg Ped Wait Time:\s*([\d\.]+)s?", content).group(1)),
            'cross_modal': float(re.search(r"FINAL CROSS-MODAL GAP:\s*([\d\.]+)s?", content).group(1)),
            'intra_lane': float(re.search(r"TRUE INTRA-LANE FAIRNESS:\s*([\d\.]+)", content).group(1)),
            'p95_ped': float(re.search(r"True 95th Percentile Ped:\s*([\d\.]+)s?", content).group(1))
        }
    except (AttributeError, ValueError):
        return None

def load_and_summarize(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not all_files: return None
    
    seed_records = []
    for f in all_files:
        parsed = parse_summary_file(f)
        if parsed is not None:
            seed_records.append(parsed)
            
    if not seed_records: return None
        
    concat_seeds = pd.DataFrame(seed_records)
    return {
        'veh_avg_mean': concat_seeds['veh_avg'].mean(), 'veh_avg_std': concat_seeds['veh_avg'].std(),
        'ped_avg_mean': concat_seeds['ped_avg'].mean(), 'ped_avg_std': concat_seeds['ped_avg'].std(),
        'cross_modal_mean': concat_seeds['cross_modal'].mean(), 'cross_modal_std': concat_seeds['cross_modal'].std(),
        'intra_lane_mean': concat_seeds['intra_lane'].mean(), 'intra_lane_std': concat_seeds['intra_lane'].std(),
        'p95_ped_mean': concat_seeds['p95_ped'].mean(), 'p95_ped_std': concat_seeds['p95_ped'].std(),
    }

print("Compiling summary statistics across all seeds and demands...")
data = {demand: {} for demand in demands}

for demand in demands:
    for model_name, folder_name in models.items():
        folder_path = os.path.join(base_dir, demand, folder_name)
        stats = load_and_summarize(folder_path)
        if stats is not None:
            data[demand][model_name] = stats

plt.style.use('bmh')
fig = plt.figure(figsize=(22, 7)) 

ax1 = plt.subplot(1, 3, 1)
x = np.arange(len(demands))
width = 0.25 

for i, model_name in enumerate(models.keys()):
    means = [data[d].get(model_name, {}).get('veh_avg_mean', 0) for d in demands]
    stds = [data[d].get(model_name, {}).get('veh_avg_std', 0) for d in demands]
    
    positions = x - width + (i * width)
    ax1.bar(positions, means, width, yerr=stds, capsize=5, label=model_name, 
            color=colors[model_name], edgecolor='black', alpha=0.9)

ax1.set_ylabel('Vehicle Average Wait Time (s)', fontweight='bold')
ax1.set_title('Macro Throughput Across Demands', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(['Low Demand', 'Regular Demand', 'High Demand'], fontweight='bold')
ax1.legend()

if "high-demand" in data and data["high-demand"]:
    model_names = list(models.keys())
    x_pos = np.arange(len(model_names))
    
    ax2 = plt.subplot(1, 3, 2)
    p95_means = [data["high-demand"][m]['p95_ped_mean'] for m in model_names]
    p95_stds = [data["high-demand"][m]['p95_ped_std'] for m in model_names]
    
    bars2 = ax2.bar(x_pos, p95_means, yerr=p95_stds, capsize=5, 
                    color=[colors[m] for m in model_names], edgecolor='black', alpha=0.8)
    
    max_y2 = max(p95_means) + max(p95_stds)
    ax2.set_ylim(0, max_y2 * 1.2)
    
    ax2.set_ylabel('P95 Ped Wait Time (s)', fontweight='bold')
    ax2.set_title('High Demand: Extreme Pedestrian Wait', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, fontweight='bold')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        error_offset = p95_stds[i] if p95_stds[i] > 0 else 0
        ax2.text(bar.get_x() + bar.get_width() / 2, height + error_offset + 1,
                 f'{height:.1f}s', ha='center', va='bottom', fontweight='bold', color='#374151')

    ax3 = plt.subplot(1, 3, 3)
    jain_means = [data["high-demand"][m]['intra_lane_mean'] for m in model_names]
    
    bars3 = ax3.bar(x_pos, jain_means, 
                    color=[colors[m] for m in model_names], edgecolor='black', alpha=0.8)
    
    ax3.set_ylabel("Jain's Fairness Index", fontweight='bold')
    ax3.set_title('High Demand: Intra-Lane Fairness', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, fontweight='bold')
    ax3.set_ylim(0, 1.15)
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold', color='#374151')

plt.tight_layout(pad=3.0)
plt.savefig('isef_final_results_board_separated.png', dpi=300, bbox_inches='tight')
print("\nSuccess! Saved separated visualizations to 'isef_final_results_board_separated.png'.")
plt.show()