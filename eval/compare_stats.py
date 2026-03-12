import pandas as pd
import matplotlib.pyplot as plt

def evaluate_and_compare(file1_path, file2_path, label1="Rule-Based Baseline", label2="AI Model"):
    print(f"Loading data from {file1_path} and {file2_path}...\n")
    
    # 1. Load the CSVs into Pandas DataFrames
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Print Summary Statistics to the Terminal
    print("-" * 50)
    print("🚦 SUMMARY STATISTICS (Averages over entire simulation) 🚦")
    print("-" * 50)
    
    # Helper to print side-by-side stats
    def print_stat(metric_name, col_name):
        val1 = df1[col_name].mean()
        val2 = df2[col_name].mean()
        diff = val1 - val2
        winner = label2 if val2 < val1 else label1
        print(f"{metric_name}:")
        print(f"  {label1}: {val1:.2f}")
        print(f"  {label2}: {val2:.2f}")
        print(f"  Winner: {winner} (by {abs(diff):.2f})\n")

    print_stat("Vehicle Average Waiting Time", 'vehicle_average_waiting_time')
    print_stat("Pedestrian Average Waiting Time", 'pedestrian_average_waiting_time')
    print_stat("Total Vehicles Stopped (Mean per step)", 'vehicle_total_stopped')

    # 3. Generate Visual Comparisons (The Graphs!)
    # We will create a single window with 3 subplots stacked on top of each other
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Traffic Model Evaluation: Rule-Based vs. AI', fontsize=16, fontweight='bold')

    # Graph A: Vehicle Average Wait Time
    axes[0].plot(df1['step'], df1['vehicle_average_waiting_time'], label=label1, color='red', linewidth=2)
    axes[0].plot(df2['step'], df2['vehicle_average_waiting_time'], label=label2, color='blue', linewidth=2)
    axes[0].set_title('Vehicle Average Waiting Time Over Time')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Graph B: Pedestrian Average Wait Time
    axes[1].plot(df1['step'], df1['pedestrian_average_waiting_time'], label=label1, color='orange', linewidth=2)
    axes[1].plot(df2['step'], df2['pedestrian_average_waiting_time'], label=label2, color='green', linewidth=2)
    axes[1].set_title('Pedestrian Average Waiting Time Over Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Graph C: Total Vehicles Stopped (Congestion Level)
    axes[2].plot(df1['step'], df1['vehicle_total_stopped'], label=label1, color='purple', linewidth=2)
    axes[2].plot(df2['step'], df2['vehicle_total_stopped'], label=label2, color='cyan', linewidth=2)
    axes[2].set_title('Total Vehicles Stopped (Gridlock/Congestion)')
    axes[2].set_xlabel('Simulation Step')
    axes[2].set_ylabel('Number of Vehicles')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # Adjust layout so labels don't overlap and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save a high-res image for your presentation board before opening the window
    plt.savefig('model_comparison_charts.png', dpi=300)
    print("Charts saved successfully as 'model_comparison_charts.png'. Opening window...")
    
    plt.show()

if __name__ == "__main__":
    # Change these filenames to match your actual CSV files!
    file_a = "rule_based_data.csv"
    file_b = "rl_model_data.csv" 
    
    evaluate_and_compare(file_a, file_b)