import subprocess
import os

num_files = 10
network_file = "grid-network.net.xml"
end_time = "3600"
period = "1.5"
fringe_factor = "10"

sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    print("Error: SUMO_HOME environment variable not found.")
    exit(1)

random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")

print(f"Starting generation of {num_files} vehicle traffic files...")

for i in range(1, num_files + 1):
    seed_value = str(i) 
    
    # Dynamically name the output file
    output_file = f"vehs_higher_seed{seed_value}.rou.xml"
    
    # Build the command for vehicles
    command = [
        "python3", random_trips_script,
        "-n", network_file,
        "-o", output_file,
        "-e", end_time,
        "-p", period,
        "--fringe-factor", fringe_factor,
        "--seed", seed_value
    ]
    
    print(f"[{i}/{num_files}] Generating {output_file} with seed {seed_value}...")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  -> Error generating {output_file}:\n{result.stderr}")
    else:
        print("  -> Success!")

print("\nAll vehicle files generated successfully!")