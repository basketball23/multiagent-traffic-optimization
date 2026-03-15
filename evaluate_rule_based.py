import os
import sys
import traci
import csv
import argparse
import numpy as np

from utils import get_intersection_metrics

MIN_GREEN_TIME = 15      
QUEUE_THRESHOLD = 3

def run_multi_rule_based():
    """
    runs the SUMO simulation using dynamic rule-based traffic lights.
    acts as actuated baseline control group.
    """

    parser = argparse.ArgumentParser(description="Run Rule-Based Baseline Evaluation")
    parser.add_argument("--net-file", type=str, required=True, help="Path to the network file")
    parser.add_argument("--route-file", type=str, required=True, help="Comma-separated route files")
    parser.add_argument("--out-csv", type=str, required=True, help="Path to save the output CSV")
    args = parser.parse_args()
    
    sumoCmd = [
        "sumo", 
        "-n", args.net_file, 
        "-r", args.route_file,
        "--no-warnings", "true"
    ]
    traci.start(sumoCmd)
    
    print(f"Running Dynamic Rule-Based Control for {args.route_file}...")
    
    step = 0
    phase_timers = {} 
    
    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            'step', 'vehicle_total_stopped', 'vehicle_total_waiting_time',
            'vehicle_average_waiting_time', 'pedestrian_total_stopped',
            'pedestrian_total_waiting_time', 'pedestrian_average_waiting_time',
            'cross_modal_fairness', 'intra_lane_fairness', 'p95_ped_wait_time'
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
                
                all_lane_waits = []
                
                '''base metrics'''
                for tl_id in tl_ids:
                    v_count, v_time, p_count, p_time = get_intersection_metrics(tl_id)
                    net_veh_count += v_count
                    net_veh_time += v_time
                    net_ped_count += p_count
                    net_ped_time += p_time

                    lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))
                    lane_waits = [traci.lane.getWaitingTime(lane) for lane in lanes]
                    all_lane_waits.extend(lane_waits)

                veh_avg_wait = net_veh_time / net_veh_count if net_veh_count > 0 else 0
                ped_avg_wait = net_ped_time / net_ped_count if net_ped_count > 0 else 0

                '''intra & cross modal fairness'''
                cross_modal_fairness = abs(ped_avg_wait - veh_avg_wait)

                sum_waits = sum(all_lane_waits)
                sum_sq_waits = sum(w**2 for w in all_lane_waits)
                n_lanes = len(all_lane_waits)
                
                if n_lanes > 0 and sum_sq_waits > 0:
                    intra_lane_fairness = (sum_waits ** 2) / (n_lanes * sum_sq_waits)
                else:
                    intra_lane_fairness = 1.0

                '''p95 wait time'''
                ped_ids = traci.person.getIDList()
                ped_waits = [traci.person.getWaitingTime(p_id) for p_id in ped_ids]
                
                if len(ped_waits) > 0:
                    p95_ped_wait = np.percentile(ped_waits, 95)
                else:
                    p95_ped_wait = 0.0

                writer.writerow([
                    step, net_veh_count, net_veh_time, veh_avg_wait,
                    net_ped_count, net_ped_time, ped_avg_wait,
                    cross_modal_fairness, intra_lane_fairness, p95_ped_wait
                ])

                file.flush()

            step += 1

    traci.close()
    print("Rule-Based Simulation Complete.")

if __name__ == "__main__":
    run_multi_rule_based()