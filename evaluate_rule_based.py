import os
import sys
import traci
import csv
import argparse
import numpy as np

from utils import get_intersection_metrics

MIN_GREEN_TIME = 15      
MAX_GREEN_TIME = 50
QUEUE_THRESHOLD = 3
PEDESTRIAN_WEIGHT = 1    

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
                
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(target_light)[0]
                num_phases = len(logic.phases)
                
                if current_phase % 2 == 0:
                    if phase_timers[target_light] >= MIN_GREEN_TIME:
                        waiting_entities_on_red = 0
                        active_entities_on_green = 0
                        
                        state_string = traci.trafficlight.getRedYellowGreenState(target_light)
                        lanes = traci.trafficlight.getControlledLanes(target_light)
                        
                        unique_red_lanes = set()
                        unique_green_lanes = set()
                        
                        for i, lane in enumerate(lanes):
                            if state_string[i].lower() in ('r', 'y'):
                                if lane not in unique_red_lanes:
                                    waiting_entities_on_red += traci.lane.getLastStepHaltingNumber(lane)
                                    unique_red_lanes.add(lane)
                            elif state_string[i].lower() in ('g', 'G'):
                                if lane not in unique_green_lanes:
                                    active_entities_on_green += traci.lane.getLastStepVehicleNumber(lane)
                                    unique_green_lanes.add(lane)
                        
                        ped_ids = traci.person.getIDList()
                        for p_id in ped_ids:
                            if traci.person.getWaitingTime(p_id) > 0:
                                next_edge = traci.person.getNextEdge(p_id)
                                if next_edge in unique_red_lanes:
                                    waiting_entities_on_red += PEDESTRIAN_WEIGHT
                        
                        demand_on_red = waiting_entities_on_red > 0
                        queue_threshold_met = waiting_entities_on_red >= QUEUE_THRESHOLD
                        green_is_empty = active_entities_on_green == 0
                        max_green_hit = phase_timers[target_light] >= MAX_GREEN_TIME
                        
                        force_switch = max_green_hit and demand_on_red
                        
                        gap_out_switch = queue_threshold_met and green_is_empty
                        
                        low_demand_switch = demand_on_red and green_is_empty
                        
                        if force_switch or gap_out_switch or low_demand_switch:
                            next_phase = (current_phase + 1) % num_phases
                            traci.trafficlight.setPhase(target_light, next_phase)
                            phase_timers[target_light] = 0 
                
                else:
                    if phase_timers[target_light] >= 4:
                        next_phase = (current_phase + 1) % num_phases
                        traci.trafficlight.setPhase(target_light, next_phase)
                        traci.trafficlight.setPhaseDuration(target_light, 1000)
                        phase_timers[target_light] = 0

            if step % 5 == 0:
                net_veh_count = 0
                net_veh_time = 0
                net_ped_count = 0
                net_ped_time = 0
                
                all_lane_waits = []
                
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

                cross_modal_fairness = abs(ped_avg_wait - veh_avg_wait)

                sum_waits = sum(all_lane_waits)
                sum_sq_waits = sum(w**2 for w in all_lane_waits)
                n_lanes = len(all_lane_waits)
                
                if n_lanes > 0 and sum_sq_waits > 0:
                    intra_lane_fairness = (sum_waits ** 2) / (n_lanes * sum_sq_waits)
                else:
                    intra_lane_fairness = 1.0

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

            step += 1

    traci.close()
    print("Rule-Based Simulation Complete.")

if __name__ == "__main__":
    run_multi_rule_based()