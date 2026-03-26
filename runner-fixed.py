import traci
import sumolib
import csv
from utils import get_intersection_metrics

NET_FILE = 'simulation/grid-network-nema.net.xml'
ROUTE_FILES = 'simulation/vehs-nema.rou.xml,simulation/peds-nema.rou.xml'
OUTPUT_FILE = 'fixed_timer_data.csv'
SIM_SECONDS = 3600
LOG_INTERVAL = 5
PROGRAM_ID = 1

def main():
    sumo_binary = sumolib.checkBinary('sumo-gui') 
    
    sumo_cmd = [
        sumo_binary,
        '-n', NET_FILE,
        '-r', ROUTE_FILES,
        '--waiting-time-memory', '10000',
        '-W'                              
    ]
    
    traci.start(sumo_cmd)
    
    tl_ids = traci.trafficlight.getIDList()

    for tl_id in tl_ids:
        try:
            traci.trafficlight.setProgram(tl_id, PROGRAM_ID)
        except traci.exceptions.TraCIException as e:
            print(f"Warning: Could not set program '{PROGRAM_ID}' on {tl_id}: {e}")
    
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        fieldnames = [
            'step', 'vehicle_total_stopped', 'vehicle_total_waiting_time',
            'vehicle_average_waiting_time', 'pedestrian_total_stopped',
            'pedestrian_total_waiting_time', 'pedestrian_average_waiting_time'
        ]
        
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in range(1, SIM_SECONDS + 1):
            traci.simulationStep()
            
            if step % LOG_INTERVAL == 0:
                net_veh_count, net_veh_time = 0, 0
                net_ped_count, net_ped_time = 0, 0

                for tl_id in tl_ids:
                    v_count, v_time, p_count, p_time = get_intersection_metrics(tl_id)
                    net_veh_count += v_count
                    net_veh_time += v_time
                    net_ped_count += p_count
                    net_ped_time += p_time

                veh_avg_wait = net_veh_time / net_veh_count if net_veh_count > 0 else 0
                ped_avg_wait = net_ped_time / net_ped_count if net_ped_count > 0 else 0

                writer.writerow({
                    'step': step,
                    'vehicle_total_stopped': net_veh_count,
                    'vehicle_total_waiting_time': net_veh_time,
                    'vehicle_average_waiting_time': round(veh_avg_wait, 2),
                    'pedestrian_total_stopped': net_ped_count,
                    'pedestrian_total_waiting_time': net_ped_time,
                    'pedestrian_average_waiting_time': round(ped_avg_wait, 2)
                })
                
    print(f"Baseline evaluation finished! Data saved to {OUTPUT_FILE}")
    traci.close()

if __name__ == "__main__":
    main()