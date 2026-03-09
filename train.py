import sumo_rl
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import traci
from collections import deque

# deque to keep track of previous traffic states to prevent rapid switching
# dictionary of deques for each intersection
BUFFER_SIZE = 10

def fair_wait_time_reward(traffic_signal):
    '''
    custom reward function to balance vehicles, pedestrians, and signal frequency switching
    '''

    sim_time = traffic_signal.sumo.simulation.getTime()

    if not hasattr(traffic_signal, 'phase_buffer') or sim_time < 5.0:
        traffic_signal.phase_buffer = deque(maxlen=BUFFER_SIZE)

        all_edges = traffic_signal.sumo.edge.getIDList()

        traffic_signal.pedestrian_edges = [e for e in all_edges if '_w' in e]

    current_phase = traffic_signal.green_phase

    # vehicle waiting and delay
    vehicle_waiting_count = traffic_signal.get_total_queued()
    vehicle_delay = sum(traffic_signal.get_accumulated_waiting_time_per_lane())

    # pedestrian delay
    pedestrian_waiting_count = 0

    # TODO: this gets you ids, now need to fetch their waiting times
    for edge in traffic_signal.pedestrian_edges:
        pedestrian_waiting_count += len(traffic_signal.sumo.edge.getLastStepPersonIDs(edge))
    
    emissions_count = traffic_signal.get_total_co2()

    # switching penalty
    switching_penalty = 0
    # switching penalty weight
    sp_w = 5.0

    if len(traffic_signal.phase_buffer) > 1:
        total_switches = 0
        history = list(traffic_signal.phase_buffer)

        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                total_switches += 1
        
        switching_penalty = sp_w * total_switches
    
    traffic_signal.phase_buffer.append(current_phase)

    # vehicle weight
    v_w = 2.0
    # pedestrian weight
    p_w = 2.0

    reward = -(vehicle_waiting_count + (v_w * vehicle_delay) + (p_w * pedestrian_waiting_count) + switching_penalty)

    return reward

def main():
    env = sumo_rl.parallel_env(
        net_file='test-network.net.xml',
        route_file='vehs.rou.xml',
        out_csv_name='results.csv',
        use_gui=True,
        num_seconds=20000,
        reward_fn=fair_wait_time_reward,
        sumo_seed=42
    )

    # used to make compatible
    env.unwrapped.render_mode = None


    # vectorization of pettingzoo to stable baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # initial proximal policy optimization (PPO) model
    alpha = 0.0003
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=3,
        learning_rate=alpha,
        gamma=0.95
    )

    model.learn(total_timesteps=100000)

    model.save("traffic_model_2")

    env.close()


if __name__ == "__main__":
    main()