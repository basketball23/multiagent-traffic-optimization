import os
import glob
import pandas as pd
from scipy import stats

base_dir = "evaluation-results/regular-demand"
folders = {
    "no_rew": os.path.join(base_dir, "marl-no-rew"),
    "main": os.path.join(base_dir, "marl")
}

def get_seed_averages(folder_path, metric):
    """Reads CSVs and returns a list of the chosen metric's average for each seed."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"Warning: No CSV files found in {folder_path}")
        return []
    
    seed_averages = []
    for file in all_files:
        df = pd.read_csv(file)
        # Get the mean of the specific metric for this simulation run
        seed_mean = df[metric].mean()
        seed_averages.append(seed_mean)
        
    return seed_averages

# The metrics we want to test
metrics_to_test = {
    "Vehicle Average Wait Time": "vehicle_average_waiting_time",
    "Pedestrian Average Wait Time": "pedestrian_average_waiting_time",
    "Cross-Modal Fairness": "cross_modal_fairness"
}

print("Running Statistical Significance Tests (Independent t-test)\n" + "-"*60)

for display_name, col_name in metrics_to_test.items():
    # Get the 10 seed averages for both models
    data_no_rew = get_seed_averages(folders["no_rew"], col_name)
    data_main = get_seed_averages(folders["main"], col_name)
    
    if not data_no_rew or not data_main:
        continue
        
    # Run independent two-sample t-test
    # We use equal_var=False (Welch's t-test) which is safer when variances might differ
    t_stat, p_value = stats.ttest_ind(data_main, data_no_rew, equal_var=False)
    
    mean_main = sum(data_main) / len(data_main)
    mean_no_rew = sum(data_no_rew) / len(data_no_rew)
    diff = mean_main - mean_no_rew
    
    print(f"Metric: {display_name}")
    print(f"  Main MARL Mean    : {mean_main:.3f}")
    print(f"  No-Reward Mean    : {mean_no_rew:.3f}")
    print(f"  Difference        : {diff:.3f}")
    print(f"  T-Statistic       : {t_stat:.3f}")
    print(f"  P-Value           : {p_value:.4f}")
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("  Conclusion        : STATISTICALLY SIGNIFICANT 🟢")
        if diff < 0: # Assuming lower is better for these specific metrics
            print("                      -> Main MARL is genuinely better.")
        else:
            print("                      -> Main MARL is genuinely worse.")
    else:
        print("  Conclusion        : NOT SIGNIFICANT 🟡")
        print("                      -> The difference is likely just statistical noise/random variance.")
    print("-" * 60)