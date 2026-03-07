import sumo_rl
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import traci
from collections import deque

# deque to keep track of previous traffic states to prevent rapid switching
# dictionary of deques for each intersection
phase_buffers = {}
BUFFER_SIZE = 10

def fair_wait_time_reward(traffic_signal):
    '''
    custom reward function to balance vehicles, pedestrians, and signal frequency switching
    '''

    ts_id = traffic_signal.id
    current_phase = traffic_signal.green_phase

    if ts_id not in phase_buffers:
        phase_buffers[ts_id] = deque(maxlen=BUFFER_SIZE)

    # vehicle delay
    vehicle_delay = traffic_signal.get_total_queued()

    # pedestrian delay
    pedestrian_waiting_count = 0
    controlled_lanes = traffic_signal.lanes
    for lane in controlled_lanes:
        edge_id = traci.lane.getEdgeID(lane)
        if "crosswalk" in edge_id or "walking" in edge_id:
            pedestrian_waiting_count += traci.edge.getLastStepHaltingNumber(edge_id)
    
    # switching penalty
    switching_penalty = 0
    # switching penality weight
    sp_w = 5.0

    if len(phase_buffers[ts_id]) > 1:
        total_switches = 0
        history = list(phase_buffers[ts_id])

        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                total_switches += 1
        
        switching_penalty = sp_w * total_switches
    
    phase_buffers[ts_id].append(current_phase)

    # pedestrian weight
    p_w = 2.0

    reward = -(vehicle_delay + (p_w * pedestrian_waiting_count) + switching_penalty)

    return reward

def main():
    env = sumo_rl.parallel_env(
        net_file='test-network.net.xml',
        route_file='vehicles.rou.xml',
        use_gui=True,
        num_seconds=3600,
        reward_fn=fair_wait_time_reward,
    )

    env.unwrapped.render_mode = None


    # vectorization of pettingzoo to stable baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=2, num_cpus=1, base_class='stable_baselines3')

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

    env.close()


if __name__ == "__main__":
    main()