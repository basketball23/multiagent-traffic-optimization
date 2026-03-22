import os
import sys
import traci
import csv
import argparse
import numpy as np
import xml.etree.ElementTree as ET

from utils import get_intersection_metrics

def parse_true_metrics(tripinfo_path):
    """Parses SUMO's native tripinfo.xml for 100% accurate wait times."""
    try:
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Warning: {tripinfo_path} not found. Returning zeros.")
        return {'veh_count': 0, 'veh_avg': 0.0, 'veh_p95': 0.0,
                'ped_count': 0, 'ped_avg': 0.0, 'ped_p95': 0.0, 'cross_modal_gap': 0.0}

    veh_waits = []
    ped_waits = []

    for trip in root.findall('tripinfo'):
        wait = float(trip.get('waitingTime', 0))
        veh_waits.append(wait)

    for person in root.findall('personinfo'):
        p_wait = 0.0
        for walk in person.findall('walk'):
            p_wait += float(walk.get('timeLoss', 0))
        ped_waits.append(p_wait)

    v_95 = np.percentile(veh_waits, 95) if veh_waits else 0.0
    v_avg = np.mean(veh_waits) if veh_waits else 0.0
    v_count = len(veh_waits)

    p_95 = np.percentile(ped_waits, 95) if ped_waits else 0.0
    p_avg = np.mean(ped_waits) if ped_waits else 0.0
    p_count = len(ped_waits)

    return {
        'veh_count': v_count, 'veh_avg': v_avg, 'veh_p95': v_95,
        'ped_count': p_count, 'ped_avg': p_avg, 'ped_p95': p_95,
        'cross_modal_gap': abs(v_avg - p_avg)
    }

def run_fixed_timer():
    parser = argparse.ArgumentParser(description="Run Fixed-Timer Baseline Evaluation")
    parser.add_argument("--net-file", type=str, required=True)
    parser.add_argument("--route-file", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
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

    print(f"Running Fixed-Timer Control for {args.route_file}...")

    step = 0

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

            for p_id in traci.person.getIDList():
                w = traci.person.getWaitingTime(p_id)
                if p_id not in ped_tracking:
                    ped_tracking[p_id] = {'last': 0, 'total': 0}
                if w > ped_tracking[p_id]['last']:
                    ped_tracking[p_id]['total'] += (w - ped_tracking[p_id]['last'])
                ped_tracking[p_id]['last'] = w

            if step % 5 == 0:
                tl_ids = traci.trafficlight.getIDList()

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

                ped_ids = traci.person.getIDList()
                ped_waits = [traci.person.getWaitingTime(p_id) for p_id in ped_ids]
                p95_ped_wait = np.percentile(ped_waits, 95) if len(ped_waits) > 0 else 0.0

                writer.writerow([
                    step, net_veh_count, net_veh_time, veh_avg_wait,
                    net_ped_count, net_ped_time, ped_avg_wait,
                    cross_modal_fairness, intra_lane_fairness, p95_ped_wait
                ])

            step += 1

    traci.close()
    print(f"Fixed-Timer Baseline Simulation Complete.")

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
        f.write(f"FIXED-TIMER Baseline Summary: {args.route_file}\n")
        f.write("="*40 + "\n")
        f.write(f"Completed Vehicle Trips:    {true_stats['veh_count']}\n")
        f.write(f"True Avg Veh Wait Time:     {true_stats['veh_avg']:.2f}s\n")
        f.write(f"True 95th Percentile Veh:   {true_stats['veh_p95']:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pedestrians Tracked:        {true_stats['ped_count']}\n")
        f.write(f"True Avg Ped Wait Time:     {true_stats['ped_avg']:.2f}s\n")
        f.write(f"True 95th Percentile Ped:   {true_stats['ped_p95']:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"FINAL CROSS-MODAL GAP:      {true_stats['cross_modal_gap']:.2f}s\n")
        f.write(f"TRUE INTRA-LANE FAIRNESS:   {true_intra_lane_fairness:.4f}\n")

    print(f"\nFinal True Metrics Summary saved to: {summary_file}")
    print("="*50)

if __name__ == "__main__":
    run_fixed_timer()