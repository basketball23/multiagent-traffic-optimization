import numpy as np
from collections import deque

from gymnasium.spaces import Box
from sumo_rl.environment.observations import ObservationFunction

phase_buffers = {}
BUFFER_SIZE = 10

NEIGHBORS_DICT = {
    'B1': ['B2', 'C1'],
    'B2': ['B1', 'C2'],
    'C1': ['B1', 'C2'],
    'C2': ['C1', 'B2'],
}

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
    lane_wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()

    vehicle_delay = sum(lane_wait_times)
    max_lane_wait_time = max(lane_wait_times) if len(lane_wait_times) > 0 else 0


    # pedestrian delay
    pedestrian_waiting_count = 0
    pedestrian_delay = 0


    for edge in traffic_signal.pedestrian_edges:
        pedestrian_ids = traffic_signal.sumo.edge.getLastStepPersonIDs(edge)
        pedestrian_waiting_count += len(pedestrian_ids)

        for p_id in pedestrian_ids:
            pedestrian_delay += traffic_signal.sumo.person.getWaitingTime(p_id)

    # switching penalty
    switching_penalty = 0
    # switching penalty weight
    sp_w = 0.8

    if len(traffic_signal.phase_buffer) > 0:

        # switching penalty only compares to previous state to prevent "credit assignment problem"
        previous_phase = traffic_signal.phase_buffer[-1]

        if current_phase != previous_phase:
            switching_penalty = sp_w
        
    traffic_signal.phase_buffer.append(current_phase)

    # weights
    v_w = 2.0
    p_w = 2.0
    f_w = 1.5


    veh_waiting_norm = vehicle_waiting_count / 10.0
    ped_wait_norm = pedestrian_waiting_count / 10.0

    veh_delay_norm = vehicle_delay / 250.0
    ped_delay_norm = pedestrian_delay / 250.0

    fairness_norm = max_lane_wait_time / 100.0

    switch_pen_norm = switching_penalty


    reward = -(veh_waiting_norm + 
               ped_delay_norm + 
               (v_w * veh_delay_norm) + 
               (p_w * ped_wait_norm) + 
               (f_w * fairness_norm) +
               switch_pen_norm)

    return reward