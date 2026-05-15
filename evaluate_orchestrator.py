import subprocess
import os

num_seeds = 10
sim_dir = "simulation/low-demand-nema"
results_dir = "evaluation-results-nema/low-demand/baseline-rule-based"
network_file = "simulation/grid-network-nema.net.xml"
marl_script = "evaluate_rule_based.py"

os.makedirs(results_dir, exist_ok=True)

print(f"Starting MARL evaluations for files in '{sim_dir}'...")

for i in range(1, num_seeds + 1):
    seed_value = str(i)
    
    veh_file = os.path.join(sim_dir, f"vehs_low_seed{seed_value}.rou.xml")
    ped_file = os.path.join(sim_dir, f"peds_low_seed{seed_value}.rou.xml")
    
    route_files_arg = f"{veh_file},{ped_file}"
    
    out_csv = os.path.join(results_dir, f"metrics_seed{seed_value}.csv")
    sumo_rl_out = os.path.join(results_dir, f"sumo_rl_seed{seed_value}")
    
    print(f"\n[{i}/{num_seeds}] Running evaluation for seed {seed_value}...")
    
    command = [
        "python3", marl_script,
        "--net-file", network_file,
        "--route-file", route_files_arg,
        "--out-csv", out_csv,
        #"--sumo-rl-out", sumo_rl_out
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  -> Error on seed {seed_value}:\n{result.stderr}")
    else:
        print(f"  -> Success! Data saved to {out_csv}")

print(f"\nAll evaluations finished! Check the '{results_dir}' folder for your CSVs.")