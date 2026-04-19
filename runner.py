import sumo_rl
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from utils import NemaPedestrianStandardizedObservation, NemaStandardizedObservation
from utils import fair_wait_time_reward, vehicle_baseline_reward, get_intersection_metrics

import traci
import csv

def main():
    # creating multi-agent environment
    env = sumo_rl.parallel_env(
        net_file='simulation/test.net.xml',
        route_file='simulation/vehs_test.rou.xml,simulation/peds_test.rou.xml',
        out_csv_name='evaluation',
        use_gui=True,
        num_seconds=3600,
        reward_fn=fair_wait_time_reward,
        observation_class=NemaPedestrianStandardizedObservation,
    )

    # used to make compatible
    env.unwrapped.render_mode = None

    env = ss.pad_observations_v0(env)

    # vectorization of pettingzoo to stable baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # TODO: Load saved env
    stats_path = "models19/vec_normalize_3300000_steps.pkl" 
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load("models19/ppo_model_3300000_steps.zip")

    with open('rl_model_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'step', 'vehicle_total_stopped', 'vehicle_total_waiting_time',
            'vehicle_average_waiting_time', 'pedestrian_total_stopped',
            'pedestrian_total_waiting_time', 'pedestrian_average_waiting_time'
        ])

        obs = env.reset()
        done = False
        
        sim_time = 0 

        while not done:
            # rl agent chooses actions
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            sim_time += 5 
            
            done = dones.any()

            tl_ids = traci.trafficlight.getIDList()
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

            veh_avg_wait = net_veh_time / net_veh_count if net_veh_count > 0 else 0
            ped_avg_wait = net_ped_time / net_ped_count if net_ped_count > 0 else 0

            # Write to CSV
            writer.writerow([
                sim_time, net_veh_count, net_veh_time, veh_avg_wait,
                net_ped_count, net_ped_time, ped_avg_wait
            ])
            file.flush()

    print("Evaluation finished! Data saved to rl_model_data.csv")
    env.close()

if __name__ == "__main__":
    main()