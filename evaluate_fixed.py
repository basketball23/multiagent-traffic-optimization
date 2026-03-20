import os
import sys
import traci
import csv
import argparse
import numpy as np

from utils import get_intersection_metrics

def run_fixed_timer():
    parser = argparse.ArgumentParser(description="Run Fixed-Timer Baseline Evaluation")
    parser.add_argument("--net-file", type=str, required=True)
    parser.add_argument("--route-file", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    sumoCmd = [
        "sumo", 
        "-n", args.net_file, 
        "-r", args.route_file,
        "--waiting-time-memory", "10000",
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

    def get_stats(tracking_dict, arrived_set=None):
        if not tracking_dict: return 0.0, 0.0, 0
        
        if arrived_set is not None:
            totals = [d['total'] for v_id, d in tracking_dict.items() if v_id in arrived_set]
        else:
            totals = [d['total'] for d in tracking_dict.values()]
            
        if not totals: return 0.0, 0.0, 0
        return np.mean(totals), np.percentile(totals, 95), len(totals)

    v_avg, v_95, v_count = get_stats(veh_tracking, arrived_vehicles)
    p_avg, p_95, p_count = get_stats(ped_tracking)
    cross_modal_gap = abs(v_avg - p_avg)

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
        f.write(f"Completed Vehicle Trips:    {v_count}\n")
        f.write(f"True Avg Veh Wait Time:     {v_avg:.2f}s\n")
        f.write(f"True 95th Percentile Veh:   {v_95:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pedestrians Tracked:        {p_count}\n")
        f.write(f"True Avg Ped Wait Time:     {p_avg:.2f}s\n")
        f.write(f"True 95th Percentile Ped:   {p_95:.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"FINAL CROSS-MODAL GAP:      {cross_modal_gap:.2f}s\n")
        f.write(f"TRUE INTRA-LANE FAIRNESS:   {true_intra_lane_fairness:.4f}\n")

    print(f"\nFinal True Metrics Summary saved to: {summary_file}")
    print("="*50)

if __name__ == "__main__":
    run_fixed_timer()