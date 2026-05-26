import subprocess
import os
import random 

num_files = 100
network_file = "grid-network-nema.net.xml"
end_time = "3600"
ped_rate = "2.5"

output_dir = "high-demand-asa"

os.makedirs(output_dir, exist_ok=True)

sumo_home = os.environ.get("SUMO_HOME")
if not sumo_home:
    print("Error: SUMO_HOME environment variable not found.")
    exit(1)

random_trips_script = "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sumo/tools/randomTrips.py"

print(f"Starting generation of {num_files} pedestrian traffic files...")

random_seeds = random.sample(range(1, 2147483648), num_files)

for i, seed_int in enumerate(random_seeds, start=1):
    seed_value = str(seed_int) 
    
    # Name the file and join it with the output directory path
    output_filename = f"peds_high_{i:02d}_seed{seed_value}.rou.xml"
    output_file = os.path.join(output_dir, output_filename)
    
    command = [
        "python3", random_trips_script,
        "-n", network_file,
        "-o", output_file,
        "-e", end_time,
        "--pedestrians",
        "-p", ped_rate,
        "--seed", seed_value
    ]
    
    print(f"[{i}/{num_files}] Generating {output_filename} in {output_dir}...")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  -> Error generating {output_filename}:\n{result.stderr}")
    else:
        print("  -> Success!")

print("\nAll pedestrian files generated successfully!")