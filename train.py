import sumo_rl
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gymnasium.spaces import Box
from sumo_rl.environment.observations import ObservationFunction

import traci
from collections import deque
import numpy as np
import os

# deque to keep track of previous traffic states to prevent rapid switching
# dictionary of deques for each intersection
BUFFER_SIZE = 10

# TODO: update with actual traffic signal neighbors
NEIGHBORS_DICT = {
    'B1': ['B2', 'C1'],
    'B2': ['B1', 'C2'],
    'C1': ['B1', 'C2'],
    'C2': ['C1', 'B2'],
}

# saving model mid-training

save_dir = "./models1"
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=save_dir,
    name_prefix="ppo_model"
)


class NeighborObservation(ObservationFunction):
    '''
    custom observation function to include observation states of neighboring intersection queue data
    to local intersections' observation state.

    goal is to make agents cooperative in this way
    '''
    def __init__(self, traffic_signal):
        '''
        initialize default observation function
        '''
        super().__init__(traffic_signal)
        self.ts = traffic_signal

        # add neighboring traffic signals
        self.neighbors = NEIGHBORS_DICT.get(self.ts.id, [])

        self.neighbor_lanes = {}
        self.neighbor_lanes_lengths = {}

        for neighbor_id in self.neighbors:
            lanes = list(dict.fromkeys(self.ts.sumo.trafficlight.getControlledLanes(neighbor_id)))
            self.neighbor_lanes[neighbor_id] = lanes

            for lane in lanes:
                self.neighbor_lanes_lengths[lane] = self.ts.sumo.lane.getLength(lane)

    def __call__(self):
        '''
        fetch observation states

        everything taken from original observation function up until neighbor_id
        '''
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        obs = phase_id + min_green + density + queue

        for neighbor_id in self.neighbors:
            for lane in self.neighbor_lanes[neighbor_id]:
                halting = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                length = self.neighbor_lanes_lengths[lane]

                normalized_queue = halting / (length / 5.0)

                obs.append(min(1.0, normalized_queue))

        obs = np.array(obs, dtype=np.float32)
        return obs

    def observation_space(self):
        '''
        return the observation space
        '''
        local_len = self.ts.num_green_phases + (2 * len(self.ts.lanes)) + 1

        neighbor_len = 0
        for neighbor_id in self.neighbors:
            neighbor_len += len(self.neighbor_lanes[neighbor_id])
        
        total_len = local_len + neighbor_len

        return Box(low=0.0, high=1.0, shape=(total_len,), dtype=np.float32)
    

def fair_wait_time_reward(traffic_signal):
    '''
    custom reward function to balance vehicles, pedestrians, and signal frequency switching

    vars:
    vehicle_delay
    pedestrain_waiting_count
    switching_penalty
    '''

    if not hasattr(traffic_signal, 'phase_buffer'):
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

    # switching penalty
    switching_penalty = 0
    # switching penalty weight
    sp_w = 0.8

    if len(traffic_signal.phase_buffer) > 1:
        total_switches = 0
        history = list(traffic_signal.phase_buffer)

        traffic_signal.phase_buffer.append(current_phase)

        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                total_switches += 1
        
        switching_penalty = sp_w * total_switches

    # vehicle weight
    v_w = 2.0
    # pedestrian weight
    p_w = 2.0


    veh_waiting_norm = vehicle_waiting_count / 10
    veh_delay_norm = vehicle_delay / 250
    ped_wait_norm = pedestrian_waiting_count / 10
    switch_pen_norm = switching_penalty / BUFFER_SIZE

    # TODO: Incorporate fairness (variance(wait_times) or max_wait_time)

    reward = -(veh_waiting_norm + (v_w * veh_delay_norm) + (p_w * ped_wait_norm) + switch_pen_norm)
    reward = np.tanh(reward)

    return reward

def main():
    '''
    1. initializes agent environment
    2. vectorizes to compatable with stable_baselines3
    3. PPO model initialization and training
    4. saves model
    '''

    # creating multi-agent environment
    env = sumo_rl.parallel_env(
        net_file='grid-network.net.xml',
        route_file='vehs.rou.xml,peds.rou.xml',
        out_csv_name='results',
        use_gui=False,
        num_seconds=20000,
        reward_fn=fair_wait_time_reward,
        observation_class=NeighborObservation,
        sumo_seed=42
    )

    # used to make compatible
    env.unwrapped.render_mode = None

    # padding to observation space necessary,
    # because not all traffic signals will have same number of neighbors
    env = ss.pad_observations_v0(env)

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
        gamma=0.95,
        #device='mps'
    )

    model.learn(total_timesteps=10000000, callback=checkpoint_callback)

    model.save("traffic_model")

    env.close()


if __name__ == "__main__":
    main()