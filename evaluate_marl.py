import argparse
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from utils import NeighborAwareObservation, fair_wait_time_reward, get_intersection_metrics 
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
        observation_class=NeighborAwareObservation,
    )

    env.unwrapped.render_mode = None
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    stats_path = "models17/vec_normalize_400000_steps.pkl" 
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load("models17/ppo_model_400000_steps.zip")

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
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            sim_time += 5 
            done = dones.any()

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

    print(f"Evaluation finished for {args.route_file}!")
    env.close()

if __name__ == "__main__":
    main()