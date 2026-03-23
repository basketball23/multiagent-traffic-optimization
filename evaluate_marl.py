import argparse
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from utils import NemaPedestrianStandardizedObservation, NemaStandardizedObservation
from utils import fair_wait_time_reward, vehicle_baseline_reward, parse_true_metrics, get_intersection_metrics

import traci
import csv
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run MARL evaluation")
    parser.add_argument("--net-file", type=str, default='simulation/grid-network.net.xml')
    parser.add_argument("--route-file", type=str, required=True, help="Comma-separated route files")
    parser.add_argument("--out-csv", type=str, required=True, help="Path to save the custom CSV")
    parser.add_argument("--sumo-rl-out", type=str, required=True, help="Prefix for sumo-rl outputs")
    args = parser.parse_args()

    env = sumo_rl.parallel_env(
        net_file=args.net_file,
        route_file=args.route_file,
        out_csv_name=args.sumo_rl_out, 
        use_gui=False, 
        num_seconds=3600,
        reward_fn=fair_wait_time_reward,
        observation_class=NemaPedestrianStandardizedObservation,
        waiting_time_memory=10000,
        sumo_seed=1,
        additional_sumo_cmd=f"--tripinfo-output {args.sumo_rl_out}_tripinfo.xml"
    )

    env.unwrapped.render_mode = None
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    stats_path = "models19/vec_normalize_330000_steps.pkl" 
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load("models19/ppo_model_3300000_steps.zip")

    veh_tracking = {}
    ped_tracking = {}
    lane_tracking = {}

    with open(args.out_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'step', 'vehicle_total_stopped', 'vehicle_total_waiting_time',
            'vehicle_average_waiting_time', 'pedestrian_total_stopped',
            'pedestrian_total_waiting_time', 'pedestrian_average_waiting_time',
            'cross_modal_fairness', 'intra_lane_fairness', 'p95_ped_wait_time'
        ])

        obs = env.reset()
        done = False
        sim_time = 0 

        while not done:
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

            '''base efficiency stats'''
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
                
                # --- NEW: Ensure all controlled lanes exist in our tracker even if 0 delay ---
                for lane in lanes:
                    if lane not in lane_tracking:
                        lane_tracking[lane] = 0.0
                
                lane_waits = [traci.lane.getWaitingTime(lane) for lane in lanes]
                all_lane_waits.extend(lane_waits)

            '''cross modal fairness using abs of averages'''
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

            '''p95 pedestrian wait time'''
            ped_ids = traci.person.getIDList()
            ped_waits = [traci.person.getWaitingTime(p_id) for p_id in ped_ids]
            
            if len(ped_waits) > 0:
                p95_ped_wait = np.percentile(ped_waits, 95)
            else:
                p95_ped_wait = 0.0

            writer.writerow([
                sim_time, net_veh_count, net_veh_time, veh_avg_wait,
                net_ped_count, net_ped_time, ped_avg_wait,
                cross_modal_fairness, intra_lane_fairness, p95_ped_wait
            ])

            if sim_time >= (3600 - 5):
                break

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            sim_time += 5
            done = dones.any()

    print(f"Evaluation finished for {args.route_file}!")
    
    env.close()

    tripinfo_file = f"{args.sumo_rl_out}_tripinfo.xml"
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
        f.write(f"MARL Evaluation Summary: {args.route_file}\n")
        f.write("="*40 + "\n")
        f.write(f"Total Vehicles Tracked:     {true_stats['veh_count']}\n")
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
    main()