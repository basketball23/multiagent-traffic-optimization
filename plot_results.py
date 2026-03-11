import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

csv_files = glob.glob('data2/*results*.csv')

latest_csv = max(csv_files, key=os.path.getctime)

df = pd.read_csv('data3/results_conn1_ep463.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

if 'system_total_waiting_time' in df.columns:
    ax1.plot(df['step'], df['system_total_waiting_time'], color='red', label='Total Wait Time')
    ax1.set_ylabel('Waiting Time (seconds)')
    ax1.set_title('Total System Waiting Time Over Time')
    ax1.grid(True)
    ax1.legend()

if 'system_total_stopped' in df.columns:
    ax2.plot(df['step'], df['system_total_stopped'], color='blue', label='Stopped Cars')
    ax2.set_xlabel('Simulation Step (Seconds)')
    ax2.set_ylabel('Number of Cars')
    ax2.set_title('Total Stopped Vehicles Over Time')
    ax2.grid(True)
    ax2.legend()

plt.tight_layout()
plt.show()