import numpy as np
from collections import deque

from gymnasium.spaces import Box
from sumo_rl.environment.observations import ObservationFunction

import traci

import os
from stable_baselines3.common.callbacks import BaseCallback

phase_buffers = {}
BUFFER_SIZE = 10

NEIGHBORS_DICT = {
    'B1': ['B2', 'C1'],
    'B2': ['B1', 'C2'],
    'C1': ['B1', 'C2'],
    'C2': ['C1', 'B2'],
}

class NeighborAwareObservation(ObservationFunction):
    '''
    Custom observation function to include observation states of neighboring intersection 
    queue data and phase states to local intersections' observation state.
    '''
    def __init__(self, traffic_signal):
        super().__init__(traffic_signal)
        self.ts = traffic_signal

        self.neighbors = NEIGHBORS_DICT.get(self.ts.id, [])

        self._setup_done = False
        self.neighbor_lanes = {}
        self.neighbor_lanes_lengths = {}

        self.neighbor_num_phases = {}

        self.local_ped_edges = []

    def _setup(self):
        '''
        Filters neighboring lanes to only include those that feed traffic
        toward the local intersection, and dynamically counts neighbor green phases.
        '''
        if self._setup_done:
            return

        local_incoming_lanes = self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)
        local_incoming_edges = set([self.ts.sumo.lane.getEdgeID(lane) for lane in local_incoming_lanes])

        for neighbor_id in self.neighbors:

            all_neighbor_lanes = list(dict.fromkeys(self.ts.sumo.trafficlight.getControlledLanes(neighbor_id)))
            
            feeding_lanes = []
            
            for lane in all_neighbor_lanes:
                links = self.ts.sumo.lane.getLinks(lane)
                
                for link in links:
                    approached_lane = link[0]
                    if approached_lane:
                        approached_edge = self.ts.sumo.lane.getEdgeID(approached_lane)
                        
                        if approached_edge in local_incoming_edges:
                            feeding_lanes.append(lane)
                            break
                            
            self.neighbor_lanes[neighbor_id] = feeding_lanes
            
            for lane in feeding_lanes:
                self.neighbor_lanes_lengths[lane] = self.ts.sumo.lane.getLength(lane)

            logic = self.ts.sumo.trafficlight.getCompleteRedYellowGreenDefinition(neighbor_id)[0]
            green_phases = [
                p for p in logic.phases 
                if "y" not in p.state and ("g" in p.state.lower() or "G" in p.state)
            ]
            self.neighbor_num_phases[neighbor_id] = len(green_phases)

        all_edges = self.ts.sumo.edge.getIDList()
        incoming_edges = set([self.ts.sumo.lane.getEdgeID(lane) for lane in self.ts.lanes])

        for e in all_edges:
            if '_w' in e or '_c' in e: 
                if self.ts.id in e or any(inc_edge in e for inc_edge in incoming_edges):
                    self.local_ped_edges.append(e)
                
        self._setup_done = True

    def __call__(self):
        '''
        Fetch observation states
        '''
        self._setup()

        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        obs = phase_id + min_green + density + queue

        '''get neighbor agent states'''

        for neighbor_id in self.neighbors:
            neighbor_ts = self.ts.env.traffic_signals[neighbor_id]
            num_phases = neighbor_ts.num_green_phases

            neighbor_phase = [1 if neighbor_ts.green_phase == i else 0 for i in range(num_phases)]

            # include neighbor phase
            obs.extend(neighbor_phase)

            for lane in self.neighbor_lanes[neighbor_id]:
                halting = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                veh_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane)
                length = self.neighbor_lanes_lengths[lane]

                normalized_queue = halting / (length / 5.0)
                normalized_density = veh_count / (length / 7.5)

                # include neighbor density and queue
                obs.append(min(1.0, normalized_queue))
                obs.append(min(1.0, normalized_density))

        '''fetch pedestrian states for observation'''
        MAX_PEDESTRIANS = 10.0 

        for edge in self.local_ped_edges:
            pedestrian_ids = self.ts.sumo.edge.getLastStepPersonIDs(edge)
            count = len(pedestrian_ids)
            
            # normalize the count to a 0.0 - 1.0 scale
            normalized_count = min(1.0, count / MAX_PEDESTRIANS)
            obs.append(normalized_count)

        return np.array(obs, dtype=np.float32)

    def observation_space(self):
        '''
        Return the dynamically sized observation space
        '''
        self._setup()
        
        local_len = self.ts.num_green_phases + (2 * len(self.ts.lanes)) + 1
        
        neighbor_len = 0
        for neighbor_id in self.neighbors:
            neighbor_len += 2 * len(self.neighbor_lanes[neighbor_id])

            neighbor_len += self.neighbor_num_phases[neighbor_id]
        
        ped_len = len(self.local_ped_edges)
        
        total_len = local_len + neighbor_len + ped_len

        return Box(low=0.0, high=1.0, shape=(total_len,), dtype=np.float32)
    

def fair_wait_time_reward(traffic_signal):
    '''
    custom reward function to balance vehicles, pedestrians, and signal frequency switching

    vars:
    vehicle_delay
    pedestrain_waiting_count
    switching_penalty
    '''

    current_sim_time = traffic_signal.sumo.simulation.getTime()
    is_new_episode = current_sim_time <= traffic_signal.env.delta_time if hasattr(traffic_signal, 'env') else current_sim_time <= 5.0

    if not hasattr(traffic_signal, 'phase_buffer') or is_new_episode:
        traffic_signal.phase_buffer = deque(maxlen=BUFFER_SIZE)

        all_edges = traffic_signal.sumo.edge.getIDList()
        incoming_edges = set([traffic_signal.sumo.lane.getEdgeID(lane) for lane in traffic_signal.lanes])

        local_ped_edges = []
        for e in all_edges:
            if '_w' in e:
                if traffic_signal.id in e or any(inc_edge in e for inc_edge in incoming_edges):
                    local_ped_edges.append(e)
                    
        traffic_signal.pedestrian_edges = local_ped_edges

    if not hasattr(traffic_signal, 'prev_vehicle_wait') or is_new_episode:
        traffic_signal.prev_vehicle_wait = deque(maxlen=1)
    if not hasattr(traffic_signal, 'prev_pedestrian_wait') or is_new_episode:
        traffic_signal.prev_pedestrian_wait = deque(maxlen=1)
    if not hasattr(traffic_signal, 'prev_max_lane') or is_new_episode:
        traffic_signal.prev_max_lane = deque(maxlen=1)
    if not hasattr(traffic_signal, 'prev_p95') or is_new_episode:
        traffic_signal.prev_p95 = deque(maxlen=1)

    current_phase = traffic_signal.green_phase

    '''vehicle waiting and delay'''

    lane_wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()

    vehicle_waiting_count = traffic_signal.get_total_queued()
    vehicle_delay = sum(lane_wait_times)
    vehicle_delay_delta = 0

    if len(traffic_signal.prev_vehicle_wait) > 0:

        # switching penalty only compares to previous state to prevent "credit assignment problem"
        previous_veh_delay = traffic_signal.prev_vehicle_wait[-1]
        raw_delay_delta = previous_veh_delay - vehicle_delay

        # cap reward to prevent buildup and release
        vehicle_delay_delta = np.clip(raw_delay_delta, -50.0, 50.0)
        
    traffic_signal.prev_vehicle_wait.append(vehicle_delay)

    '''max lane wait time'''
    max_lane_wait_time = max(lane_wait_times) if len(lane_wait_times) > 0 else 0
    max_lane_wait_delta = 0

    if len(traffic_signal.prev_max_lane) > 0:
        previous_max_lane = traffic_signal.prev_max_lane[-1]
        max_lane_wait_delta = previous_max_lane - max_lane_wait_time
    
    traffic_signal.prev_max_lane.append(max_lane_wait_time)

    '''pedestrian delay'''

    pedestrian_waiting_count = 0
    pedestrian_delay = 0
    pedestrian_delay_delta = 0
    ped_wait_times = []

    for edge in traffic_signal.pedestrian_edges:
        pedestrian_ids = traffic_signal.sumo.edge.getLastStepPersonIDs(edge)
        pedestrian_waiting_count += len(pedestrian_ids)

        for p_id in pedestrian_ids:
            p_wait = traffic_signal.sumo.person.getWaitingTime(p_id)
            pedestrian_delay += p_wait
            ped_wait_times.append(p_wait)

    if len(traffic_signal.prev_pedestrian_wait) > 0:
        # switching penalty only compares to previous state to prevent "credit assignment problem"
        previous_ped_delay = traffic_signal.prev_pedestrian_wait[-1]
        pedestrian_delay_delta = previous_ped_delay - pedestrian_delay
        
    traffic_signal.prev_pedestrian_wait.append(pedestrian_delay)

    '''p95 calculation'''

    p95_ped_wait = 0
    if len(ped_wait_times) > 0:
        p95_ped_wait = np.percentile(ped_wait_times, 95)
    
    p95_ped_delta = 0
    if len(traffic_signal.prev_p95) > 0:
        previous_p95 = traffic_signal.prev_p95[-1]
        p95_ped_delta = previous_p95 - p95_ped_wait
    
    traffic_signal.prev_p95.append(p95_ped_wait)

    '''switching penalty'''

    switching_penalty = 0
    # switching penalty weight
    sp_w = 0.8

    if len(traffic_signal.phase_buffer) > 0:

        # switching penalty only compares to previous state to prevent "credit assignment problem"
        previous_phase = traffic_signal.phase_buffer[-1]

        if current_phase != previous_phase:
            switching_penalty = sp_w
        
    traffic_signal.phase_buffer.append(current_phase)


    avg_veh_wait = vehicle_delay / vehicle_waiting_count if vehicle_waiting_count > 0 else 0
    avg_ped_wait = pedestrian_delay / pedestrian_waiting_count if pedestrian_waiting_count > 0 else 0

    mode_equity = abs(avg_veh_wait - avg_ped_wait)

    # weights
    v_w = 2.0 # vehicle waiting time
    f_w = 1.5 # fairness for max waiting time
    e_w = 1.0 # equity from vehs to peds weight
    t_w = 1.2 # tail pedestrian wait time (95th percentile)

    veh_delay_norm = vehicle_delay_delta / 10.0
    ped_delay_norm = pedestrian_delay_delta / 10.0

    fairness_norm = max_lane_wait_delta / 10.0
    p95_ped_norm = p95_ped_delta / 10.0

    switch_pen_norm = switching_penalty
    equity_norm = min(mode_equity / 10.0, 5)


    reward = (ped_delay_norm 
              + (v_w * veh_delay_norm)  
              + (f_w * fairness_norm) 
              + (t_w * p95_ped_norm) 
              - (e_w * equity_norm) 
              - switch_pen_norm
              )

    return reward


def get_intersection_metrics(tl_id):
    '''
    Helper function to extract wait times and counts for an intersection.
    '''
    veh_wait_count = 0
    veh_wait_time = 0
    ped_wait_count = 0
    ped_wait_time = 0

    lanes = set(traci.trafficlight.getControlledLanes(tl_id))
    
    for lane in lanes:
        edge_id = traci.lane.getEdgeID(lane)
        if "_w" in edge_id:
            pedestrian_ids = traci.edge.getLastStepPersonIDs(edge_id)
            ped_wait_count += len(pedestrian_ids)

            for p_id in pedestrian_ids:
                ped_wait_time += traci.person.getWaitingTime(p_id)
        else:
            veh_wait_count += traci.lane.getLastStepHaltingNumber(lane)
            veh_wait_time += traci.lane.getWaitingTime(lane)
            
    return veh_wait_count, veh_wait_time, ped_wait_count, ped_wait_time


class SaveVecNormalizeCallback(BaseCallback):
    '''
    Custom callback for saving a model and its VecNormalize statistics at the same time.
    '''
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ppo_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Construct file paths
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            stats_path = os.path.join(self.save_path, f"vec_normalize_{self.num_timesteps}_steps.pkl")
            
            self.model.save(model_path)
            
            if hasattr(self.training_env, 'save'):
                self.training_env.save(stats_path)
            
            if self.verbose > 0:
                print(f"\nSaved model checkpoint to {model_path}.zip")
                print(f"Saved VecNormalize stats to {stats_path}")
                
        return True