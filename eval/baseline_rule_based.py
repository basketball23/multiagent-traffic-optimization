import os
import sys
import traci
import csv

from utils import get_intersection_metrics

MIN_GREEN_TIME = 15      
QUEUE_THRESHOLD = 3

def run_multi_rule_based():
    sumoCmd = ["sumo", "-c", "simulation/sim.sumocfg"]
    traci.start(sumoCmd)
    
    print("Running Dynamic Rule-Based Control and logging data...")
    
    step = 0
    phase_timers = {} 
    
    with open('rule_based_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            'step',
            'vehicle_total_stopped',
            'vehicle_total_waiting_time',
            'vehicle_average_waiting_time',
            'pedestrian_total_stopped',
            'pedestrian_total_waiting_time',
            'pedestrian_average_waiting_time'
        ])
        
        while step < 3600:
            traci.simulationStep()
            tl_ids = traci.trafficlight.getIDList()
            
            for target_light in tl_ids:
                if target_light not in phase_timers:
                    phase_timers[target_light] = 0
                    traci.trafficlight.setPhaseDuration(target_light, 1000)
                    
                phase_timers[target_light] += 1
                current_phase = traci.trafficlight.getPhase(target_light)
                
                # Green phases
                if current_phase % 2 == 0:

                    if phase_timers[target_light] >= MIN_GREEN_TIME:

                        waiting_entities = 0
                        
                        state_string = traci.trafficlight.getRedYellowGreenState(target_light)
                        lanes = traci.trafficlight.getControlledLanes(target_light)
                        
                        for i, lane in enumerate(lanes):
                            if state_string[i].lower() == 'r':
                                waiting_entities += traci.lane.getLastStepHaltingNumber(lane)
                        
                        if waiting_entities >= QUEUE_THRESHOLD:

                            next_phase = (current_phase + 1) % 4
                            traci.trafficlight.setPhase(target_light, next_phase)
                            phase_timers[target_light] = 0 
                
                # Yellow phase
                else:

                    if phase_timers[target_light] >= 4:

                        next_phase = (current_phase + 1) % 4
                        traci.trafficlight.setPhase(target_light, next_phase)

                        traci.trafficlight.setPhaseDuration(target_light, 1000)
                        phase_timers[target_light] = 0


            if step % 5 == 0:

                net_veh_count = 0
                net_veh_time = 0
                net_ped_count = 0
                net_ped_time = 0
                
                for tl_id in tl_ids:

                    v_count, v_time, p_count, p_time = get_intersection_metrics(tl_id)

                    net_veh_count += v_count
                    net_veh_time += v_time
                    net_ped_count += p_count
                    net_ped_time += p_time

                # Compute averages safely
                veh_avg_wait = net_veh_time / net_veh_count if net_veh_count > 0 else 0
                ped_avg_wait = net_ped_time / net_ped_count if net_ped_count > 0 else 0

                writer.writerow([
                    step,
                    net_veh_count,
                    net_veh_time,
                    veh_avg_wait,
                    net_ped_count,
                    net_ped_time,
                    ped_avg_wait
                ])

                file.flush()

            step += 1

    traci.close()

    print("Rule-Based Simulation Complete. Data saved to rule_based_data.csv.")


if __name__ == "__main__":
    run_multi_rule_based()