import subprocess
import os

num_files = 10
network_file = "grid-network-nema.net.xml"
end_time = "3600"
ped_rate = "2.5"

sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    print("Error: SUMO_HOME environment variable not found.")
    exit(1)

random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")

print(f"Starting generation of {num_files} traffic files...")

for i in range(1, num_files + 1):
    seed_value = str(i) 
    
    output_file = f"peds_high_seed{seed_value}.rou.xml"
    
    command = [
        "python3", random_trips_script,
        "-n", network_file,
        "-o", output_file,
        "-e", end_time,
        "--pedestrians",
        "-p", ped_rate,
        "--seed", seed_value
    ]
    
    print(f"[{i}/{num_files}] Generating {output_file} with seed {seed_value}...")
    
    # Run the command in the terminal
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"  -> Error generating {output_file}:\n{result.stderr}")
    else:
        print("  -> Success!")

print("\nAll traffic files generated successfully!")