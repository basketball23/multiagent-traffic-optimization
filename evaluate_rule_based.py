import traci
import csv
import argparse
import numpy as np

from utils import parse_true_metrics, get_intersection_metrics

MIN_GREEN_TIME = 8
MAX_GREEN_TIME = 35
QUEUE_THRESHOLD = 3
PEDESTRIAN_WEIGHT = 10   
MAX_PED_WAIT_ALLOWANCE = 45
PROGRAM_ID = 1

def run_multi_rule_based():
    """
    Runs the SUMO simulation using dynamic rule-based traffic lights.
    Acts as actuated baseline control group.
    """

    parser = argparse.ArgumentParser(description="Run Rule-Based Baseline Evaluation")
    parser.add_argument("--net-file", type=str, required=True, help="Path to the network file")
    parser.add_argument("--route-file", type=str, required=True, help="Comma-separated route files")
    parser.add_argument("--out-csv", type=str, required=True, help="Path to save the output CSV")
    args = parser.parse_args()
    
    tripinfo_file = args.out_csv.replace(".csv", "_tripinfo.xml")

    sumoCmd = [
        "sumo", 
        "-n", args.net_file, 
        "-r", args.route_file,
        "--waiting-time-memory", "10000",
        "--tripinfo-output", tripinfo_file,
        "--no-warnings", "true"
    ]
    traci.start(sumoCmd)
    
    print(f"Running Dynamic Rule-Based Control for {args.route_file}...")

    for tl_id in traci.trafficlight.getIDList():
        traci.trafficlight.setProgram(tl_id, str(PROGRAM_ID))
    
    step = 0
    phase_timers = {} 
    previous_phases = {}
    
    veh_tracking = {}
    ped_tracking = {}
    lane_tracking = {}
    
    arrived_vehicles = set()
    
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
            arrived_vehicles.update(traci.simulation.getArrivedIDList())
            
            # --- VEHICLE TRACKING ---
            for v_id in traci.vehicle.getIDList():
                w = traci.vehicle.getAccumulatedWaitingTime(v_id)
                if v_id not in veh_tracking:
                    veh_tracking[v_id] = {'last': 0, 'total': 0}
                if w > veh_tracking[v_id]['last']:
                    delta = w - veh_tracking[v_id]['last']
                    veh_tracking[v_id]['total'] += delta
                    
                    lane_id = traci.vehicle.getLaneID(v_id)
                    if lane_id not in lane_tracking:
                        lane_tracking[lane_id] = 0.0
                    lane_tracking[lane_id] += delta
                veh_tracking[v_id]['last'] = w

            # --- PEDESTRIAN TRACKING ---
            for p_id in traci.person.getIDList():
                w = traci.person.getWaitingTime(p_id)
                if p_id not in ped_tracking:
                    ped_tracking[p_id] = {'last': 0, 'total': 0}
                if w > ped_tracking[p_id]['last']:
                    ped_tracking[p_id]['total'] += (w - ped_tracking[p_id]['last'])
                ped_tracking[p_id]['last'] = w

            # --- ACTUATED LOGIC ---
            tl_ids = traci.trafficlight.getIDList()
            
            for target_light in tl_ids:
                current_phase = traci.trafficlight.getPhase(target_light)
                state_string = traci.trafficlight.getRedYellowGreenState(target_light)

                # Initialize tracker
                if target_light not in previous_phases:
                    previous_phases[target_light] = current_phase
                    phase_timers[target_light] = 0

                # Detect Phase Transitions
                if current_phase != previous_phases[target_light]:
                    previous_phases[target_light] = current_phase
                    phase_timers[target_light] = 0
                
                phase_timers[target_light] += 1

                # ACTUATED LOGIC: Only run if the state string has a PRIORITY GREEN 'G'
                # This is more reliable than current_phase % 3 if your XML is inconsistent.
                if 'G' in state_string:
                    # Force the green light to stay on by default
                    traci.trafficlight.setPhaseDuration(target_light, 1000)

                    if phase_timers[target_light] >= MIN_GREEN_TIME:
                        waiting_on_red = 0
                        active_on_green = 0
                        
                        links = traci.trafficlight.getControlledLinks(target_light)
                        active_lanes = set()
                        red_lanes = set()

                        for i, state in enumerate(state_string):
                            if i < len(links) and links[i]:
                                for connection in links[i]:
                                    lane = connection[0]
                                    if state == 'G':
                                        active_lanes.add(lane)
                                    else:
                                        red_lanes.add(lane)
                        
                        # Use Speed Threshold (only count cars moving > 2m/s as "Active")
                        for lane in active_lanes:
                            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                            for v_id in veh_ids:
                                # Distance check + Speed check (prevents stalled cars from holding green)
                                dist_to_inter = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(v_id)
                                if dist_to_inter < 60 and traci.vehicle.getSpeed(v_id) > 2.0:
                                    active_on_green += 1

                        # Count waiting cars on red lanes
                        for lane in red_lanes:
                            waiting_on_red += traci.lane.getLastStepHaltingNumber(lane)

                        # Pedestrian Override
                        ped_waiting = False
                        tls_edges = set(l.split('_')[0] for l in list(active_lanes) + list(red_lanes))
                        for p_id in traci.person.getIDList():
                            if traci.person.getWaitingTime(p_id) > MAX_PED_WAIT_ALLOWANCE:
                                if traci.person.getRoadID(p_id) in tls_edges:
                                    ped_waiting = True
                                    break

                        # DECISION
                        max_green_reached = phase_timers[target_light] >= MAX_GREEN_TIME
                        no_more_flow = active_on_green == 0
                        heavy_waiting = waiting_on_red >= QUEUE_THRESHOLD

                        if (max_green_reached) or (no_more_flow and heavy_waiting) or ped_waiting:
                            # Advance to Yellow
                            num_phases = len(traci.trafficlight.getAllProgramLogics(target_light)[0].phases)
                            traci.trafficlight.setPhase(target_light, (current_phase + 1) % num_phases)
                else:
                    # During Yellow/Red, let SUMO's natural durations take over
                    # We do NOT call setPhaseDuration(1000) here.
                    pass

            # --- STATISTICS LOGGING ---
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
                    
                    for lane in lanes:
                        if lane not in lane_tracking:
                            lane_tracking[lane] = 0.0
                            
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

                ped_waits = [traci.person.getWaitingTime(p_id) for p_id in traci.person.getIDList()]
                p95_ped_wait = np.percentile(ped_waits, 95) if ped_waits else 0.0

                writer.writerow([
                    step, net_veh_count, net_veh_time, veh_avg_wait,
                    net_ped_count, net_ped_time, ped_avg_wait,
                    cross_modal_fairness, intra_lane_fairness, p95_ped_wait
                ])

            step += 1

    traci.close()
    print("Rule-Based Simulation Complete.")

    # --- FINAL SUMMARY REPORTING ---
    true_stats = parse_true_metrics(tripinfo_file)

    lane_delays = list(lane_tracking.values())
    if len(lane_delays) > 0 and sum(d**2 for d in lane_delays) > 0:
        s_w = sum(lane_delays)
        s_sq_w = sum(d**2 for d in lane_delays)
        true_intra_lane_fairness = (s_w ** 2) / (len(lane_delays) * s_sq_w)
    else:
        true_intra_lane_fairness = 1.0

    summary_file = args.out_csv.replace(".csv", "_FINAL_SUMMARY.txt")
    with open(summary_file, "w") as f:
        f.write(f"RULE-BASED Baseline Summary: {args.route_file}\n")
        f.write("="*40 + "\n")
        f.write(f"Completed Vehicle Trips:    {true_stats['veh_count']}\n")
        f.write(f"True Avg Veh Wait Time:     {true_stats['veh_avg']:.2f}s\n")
        f.write(f"True 95th Percentile Veh:   {true_stats['veh_p95']:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Pedestrians Tracked:  {true_stats['ped_count']}\n")
        f.write(f"True Avg Ped Wait Time:     {true_stats['ped_avg']:.2f}s\n")
        f.write(f"True 95th Percentile Ped:   {true_stats['ped_p95']:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"FINAL CROSS-MODAL GAP:      {true_stats['cross_modal_gap']:.2f}s\n")
        f.write(f"TRUE INTRA-LANE FAIRNESS:   {true_intra_lane_fairness:.4f}\n") 

    print(f"\nFinal True Metrics Summary saved to: {summary_file}")
    print("="*50)

if __name__ == "__main__":
    run_multi_rule_based()