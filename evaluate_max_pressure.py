import os
import sys
import traci
import csv
import argparse
import numpy as np

from utils import get_intersection_metrics

MIN_GREEN_TIME = 15      

def calculate_phase_pressure(state_string, links):
    """
    Calculates the pressure of a specific phase based on incoming vs outgoing queues.
    Pressure = Sum(Queue_in - Queue_out) for all green links in the phase.
    """
    pressure = 0
    processed_lanes = set()
    
    for i, char in enumerate(state_string):
        if char.lower() == 'g':
            if i < len(links) and links[i]:
                in_lane = links[i][0][0]
                out_lane = links[i][0][1]
                
                if (in_lane, out_lane) not in processed_lanes:
                    q_in = traci.lane.getLastStepHaltingNumber(in_lane)
                    q_out = traci.lane.getLastStepHaltingNumber(out_lane)
                    
                    pressure += (q_in - q_out)
                    processed_lanes.add((in_lane, out_lane))
                    
    return pressure

def run_max_pressure_baseline():
    """
    runs the SUMO simulation using Max-Pressure traffic light control
    acts as a throughput-maximizing baseline control group
    """

    parser = argparse.ArgumentParser(description="Run Max-Pressure Baseline Evaluation")
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
    
    print(f"Running Max-Pressure Control for {args.route_file}...")
    
    step = 0
    phase_timers = {}
    target_green_phases = {}
    
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
                    target_green_phases[target_light] = None
                    traci.trafficlight.setPhaseDuration(target_light, 1000)
                    
                phase_timers[target_light] += 1
                current_phase = traci.trafficlight.getPhase(target_light)
                
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(target_light)[0]
                num_phases = len(logic.phases)
                links = traci.trafficlight.getControlledLinks(target_light)
                
                if current_phase % 2 == 0:  # Currently in a Green Phase
                    if phase_timers[target_light] >= MIN_GREEN_TIME:
                        
                        current_state = traci.trafficlight.getRedYellowGreenState(target_light)
                        current_pressure = calculate_phase_pressure(current_state, links)
                        
                        max_other_pressure = current_pressure # Baseline to beat
                        best_phase = current_phase            # Default to staying here
                        
                        # Find the actual highest pressure phase
                        for p_idx, phase in enumerate(logic.phases):
                            if p_idx % 2 == 0 and p_idx != current_phase:
                                p_pressure = calculate_phase_pressure(phase.state, links)
                                if p_pressure > max_other_pressure:
                                    max_other_pressure = p_pressure
                                    best_phase = p_idx
                        
                        # If a different phase has higher pressure, initiate switch
                        if best_phase != current_phase:
                            target_green_phases[target_light] = best_phase
                            traci.trafficlight.setPhase(target_light, current_phase + 1) # Switch to yellow
                            phase_timers[target_light] = 0 
                
                else:  # Currently in a Yellow Phase
                    if phase_timers[target_light] >= 4:
                        # Fetch the targeted green phase, fallback to sequential if none exists
                        next_phase = target_green_phases[target_light] if target_green_phases[target_light] is not None else (current_phase + 1) % num_phases
                        
                        traci.trafficlight.setPhase(target_light, next_phase)
                        traci.trafficlight.setPhaseDuration(target_light, 1000)
                        
                        phase_timers[target_light] = 0
                        target_green_phases[target_light] = None # Reset target

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

            step += 1

    traci.close()
    print("Max-Pressure Simulation Complete.")

if __name__ == "__main__":
    run_max_pressure_baseline()