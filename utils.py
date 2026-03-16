import numpy as np

from gymnasium.spaces import Box
from sumo_rl.environment.observations import ObservationFunction

import traci

import os
from stable_baselines3.common.callbacks import BaseCallback


NEIGHBORS_DICT = {
    'B1': ['B2', 'C1'],
    'B2': ['B1', 'C2'],
    'C1': ['B1', 'C2'],
    'C2': ['C1', 'B2'],
}

class NeighborAwareObservation(ObservationFunction):
    '''
    custom observation function to include observation states of neighboring intersection 
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
        filters neighboring lanes to only include those that feed traffic
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
        fetch observation states
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
        return the dynamically sized observation space
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
    

def jains_fairness_index(values):
    """
    returns a value from 1/n (unfair) to 1.0 (perfectly fair)
    """
    if not values or sum(values) == 0:
        return 1.0
    n = len(values)
    return (sum(values)**2) / (n * sum(v**2 for v in values))

def fair_wait_time_reward(traffic_signal):
    current_sim_time = traffic_signal.sumo.simulation.getTime()
    delta_t = traffic_signal.env.delta_time if hasattr(traffic_signal, 'env') else 5.0
    is_new_episode = current_sim_time <= delta_t

    if not hasattr(traffic_signal, 'prev_stats') or is_new_episode:
        all_edges = traffic_signal.sumo.edge.getIDList()
        
        incoming_edges = set([traffic_signal.sumo.lane.getEdgeID(lane) for lane in traffic_signal.lanes])
        traffic_signal.incoming_edges = list(incoming_edges)
        
        out_edges = set()
        for lane in traffic_signal.lanes:
            for link in traffic_signal.sumo.lane.getLinks(lane):
                if link[0]:
                    out_edges.add(traffic_signal.sumo.lane.getEdgeID(link[0]))
        traffic_signal.outgoing_edges = list(out_edges)

        traffic_signal.pedestrian_edges = [e for e in all_edges if '_w' in e and 
                                           (traffic_signal.id in e or any(inc in e for inc in incoming_edges))]
        
        traffic_signal.prev_stats = {
            'veh_delay': 0,
            'ped_delay': 0,
            'max_lane_wait': 0,
            'p95_ped': 0,
            'equity_idx': 1.0,
            'ema_veh': 0.0,
            'ema_ped': 0.0,
            'pressure': 0.0,
            'phase': None
        }


    '''vehicle metrics'''
    lane_wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    vehicle_waiting_count = traffic_signal.get_total_queued()
    vehicle_delay = sum(lane_wait_times)

    '''max lane wait'''
    max_lane_wait = max(lane_wait_times) if lane_wait_times else 0

    '''pedestrian metrics'''
    ped_wait_times = []
    for edge in traffic_signal.pedestrian_edges:
        p_ids = traffic_signal.sumo.edge.getLastStepPersonIDs(edge)
        for p_id in p_ids:
            ped_wait_times.append(traffic_signal.sumo.person.getWaitingTime(p_id))
    
    pedestrian_delay = sum(ped_wait_times)
    p95_ped_wait = np.percentile(ped_wait_times, 95) if ped_wait_times else 0

    '''averages and EMA Smoothing'''
    avg_veh = vehicle_delay / vehicle_waiting_count if vehicle_waiting_count > 0 else 0.0
    avg_ped = pedestrian_delay / len(ped_wait_times) if ped_wait_times else 0.0

    # smooth the averages so a sudden burst doesn't explode the gradients
    alpha = 0.2
    ema_veh = (1 - alpha) * traffic_signal.prev_stats['ema_veh'] + (alpha * avg_veh)
    ema_ped = (1 - alpha) * traffic_signal.prev_stats['ema_ped'] + (alpha * avg_ped)

    '''fairness/equity metrics'''
    # if a group is empty don't penalize
    if vehicle_waiting_count == 0 or len(ped_wait_times) == 0:
        current_equity_idx = traffic_signal.prev_stats['equity_idx']
    else:
        current_equity_idx = jains_fairness_index([ema_veh, ema_ped])

    '''pressure calculation'''
    # total vehicles approaching vs total vehicles departing
    in_count = sum(traffic_signal.sumo.edge.getLastStepVehicleNumber(e) for e in traffic_signal.incoming_edges)
    out_count = sum(traffic_signal.sumo.edge.getLastStepVehicleNumber(e) for e in traffic_signal.outgoing_edges)
    current_pressure = abs(in_count - out_count)

    '''calculate and clip deltas'''
    # clip to prevent massive reward spikes
    v_delay_delta = np.clip(traffic_signal.prev_stats['veh_delay'] - vehicle_delay, -50, 50)
    p_delay_delta = np.clip(traffic_signal.prev_stats['ped_delay'] - pedestrian_delay, -50, 50)
    max_lane_delta = np.clip(traffic_signal.prev_stats['max_lane_wait'] - max_lane_wait, -20, 20)
    p95_delta = np.clip(traffic_signal.prev_stats['p95_ped'] - p95_ped_wait, -20, 20)
    equity_delta = current_equity_idx - traffic_signal.prev_stats['equity_idx']
    pressure_delta = np.clip(traffic_signal.prev_stats['pressure'] - current_pressure, -30, 30)

    '''switching penalty'''
    current_phase = traffic_signal.green_phase
    switching_penalty = 1.0 if (traffic_signal.prev_stats['phase'] is not None and 
                               current_phase != traffic_signal.prev_stats['phase']) else 0.0

    w_veh = 2.0
    w_ped = 1.5
    w_fair = 1.0
    w_equity = 3.0
    w_switch = 0.8
    w_pressure = 1.5

    reward = (
        (w_veh * (v_delay_delta / 20.0)) + 
        (w_ped * (p_delay_delta / 20.0)) + 
        (w_fair * (max_lane_delta / 10.0)) + 
        (w_fair * (p95_delta / 10.0)) +
        (w_equity * equity_delta) +
        (w_pressure * (pressure_delta / 15.0)) -
        (w_switch * switching_penalty)
    )

    # final overall reward clipping to keep the critic stable
    reward = np.clip(reward, -15.0, 15.0)

    traffic_signal.prev_stats = {
        'veh_delay': vehicle_delay,
        'ped_delay': pedestrian_delay,
        'max_lane_wait': max_lane_wait,
        'p95_ped': p95_ped_wait,
        'equity_idx': current_equity_idx,
        'ema_veh': ema_veh,
        'ema_ped': ema_ped,
        'pressure': current_pressure,
        'phase': current_phase
    }

    return reward


def vehicle_baseline_reward(traffic_signal):
    """
    optimized vehicle-only reward function for MARL baseline.
    computes reward based strictly on vehicle delay, max lane wait, and phase switching.
    """
    current_sim_time = traffic_signal.sumo.simulation.getTime()
    
    delta_t = traffic_signal.env.delta_time if hasattr(traffic_signal, 'env') else 5.0
    is_new_episode = current_sim_time <= delta_t

    if not hasattr(traffic_signal, 'prev_stats') or is_new_episode:        
        incoming_edges = set([traffic_signal.sumo.lane.getEdgeID(lane) for lane in traffic_signal.lanes])
        traffic_signal.incoming_edges = list(incoming_edges)
        
        out_edges = set()
        for lane in traffic_signal.lanes:
            for link in traffic_signal.sumo.lane.getLinks(lane):
                if link[0]:
                    out_edges.add(traffic_signal.sumo.lane.getEdgeID(link[0]))
        traffic_signal.outgoing_edges = list(out_edges)

        traffic_signal.prev_stats = {
            'veh_delay': 0,
            'max_lane_wait': 0,
            'phase': None,
            'pressure': 0.0
        }


    lane_wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    vehicle_delay = sum(lane_wait_times)
    max_lane_wait = max(lane_wait_times) if lane_wait_times else 0


    in_count = sum(traffic_signal.sumo.edge.getLastStepVehicleNumber(e) for e in traffic_signal.incoming_edges)
    out_count = sum(traffic_signal.sumo.edge.getLastStepVehicleNumber(e) for e in traffic_signal.outgoing_edges)
    current_pressure = abs(in_count - out_count)

    v_delay_delta = traffic_signal.prev_stats['veh_delay'] - vehicle_delay
    max_lane_delta = traffic_signal.prev_stats['max_lane_wait'] - max_lane_wait
    pressure_delta = np.clip(traffic_signal.prev_stats['pressure'] - current_pressure, -30, 30)


    current_phase = traffic_signal.green_phase
    switching_penalty = 1.0 if (traffic_signal.prev_stats['phase'] is not None and 
                               current_phase != traffic_signal.prev_stats['phase']) else 0.0

    w_veh = 2.0
    w_max_wait = 1.2
    w_pressure = 1.5
    w_switch = 0.8

    reward = (
        (w_veh * (v_delay_delta / 20.0)) + 
        (w_max_wait * (max_lane_delta / 10.0)) + 
        (w_pressure * (pressure_delta / 15.0)) -
        (w_switch * switching_penalty)
    )

    traffic_signal.prev_stats = {
        'veh_delay': vehicle_delay,
        'max_lane_wait': max_lane_wait,
        'phase': current_phase,
        'pressure': current_pressure
    }

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