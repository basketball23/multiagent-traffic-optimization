import numpy as np
import math

from gymnasium.spaces import Box
from sumo_rl.environment.observations import ObservationFunction

import traci

import os
from stable_baselines3.common.callbacks import BaseCallback

import xml.etree.ElementTree as ET

'''
OBSERVATION CLASSES
'''

class NemaPedestrianStandardizedObservation(ObservationFunction):
    def __init__(self, traffic_signal):
        super().__init__(traffic_signal)
        self.ts = traffic_signal
        self._setup_done = False
        
        # Vehicles: Lanes mapped to 8 standard indices
        # Indices: 0:NB_SR, 1:NB_L, 2:EB_SR, 3:EB_L, 4:SB_SR, 5:SB_L, 6:WB_SR, 7:WB_L
        self.movement_lanes = {i: [] for i in range(8)}
        
        # Pedestrians: Unique standard edges mapped to 4 approaches
        # Indices: 0:North, 1:East, 2:South, 3:West
        self.pedestrian_edges = {i: set() for i in range(4)}

    def _get_bearing(self, lane_id):
        """Calculates the compass bearing of an incoming edge pointing toward the junction."""
        shape = self.ts.sumo.lane.getShape(lane_id)
        x1, y1 = shape[-2][:2]
        x2, y2 = shape[-1][:2]
        
        angle_rad = math.atan2(x2 - x1, y2 - y1)
        bearing = (math.degrees(angle_rad) + 360) % 360
        return bearing

    def _determine_approach_direction(self, bearing):
        """Maps a compass bearing to an Approach Direction (N, E, S, W)."""
        if 135 <= bearing < 225:
            return "N" 
        elif 225 <= bearing < 315:
            return "E" 
        elif 315 <= bearing <= 360 or 0 <= bearing < 45:
            return "S" 
        elif 45 <= bearing < 135:
            return "W" 

    def _setup(self):
        """Maps arbitrary SUMO lanes and edges to the standard movements/approaches."""
        if self._setup_done:
            return

        incoming_lanes = self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)
        incoming_lanes = list(dict.fromkeys(incoming_lanes))

        for lane in incoming_lanes:
            edge_id = self.ts.sumo.lane.getEdgeID(lane)
            bearing = self._get_bearing(lane)
            approach_dir = self._determine_approach_direction(bearing)

            links = self.ts.sumo.lane.getLinks(lane)
            is_left_turn = False
            
            for link in links:
                if link[2] in ['l', 'L']:
                    is_left_turn = True
                    break

            if approach_dir == "N":
                idx = 1 if is_left_turn else 0
                ped_idx = 0
            elif approach_dir == "E":
                idx = 3 if is_left_turn else 2
                ped_idx = 1
            elif approach_dir == "S":
                idx = 5 if is_left_turn else 4
                ped_idx = 2
            elif approach_dir == "W":
                idx = 7 if is_left_turn else 6
                ped_idx = 3

            self.movement_lanes[idx].append(lane)
            
            if not edge_id.startswith(':'):
                self.pedestrian_edges[ped_idx].add(edge_id)

        self._setup_done = True

    def __call__(self):
        """Fetches the standardized observation state, now including pedestrians."""
        self._setup()

        standardized_density = np.zeros(8, dtype=np.float32)
        standardized_queue = np.zeros(8, dtype=np.float32)
        standardized_peds = np.zeros(4, dtype=np.float32)

        for movement_idx in range(8):
            lanes = self.movement_lanes[movement_idx]
            
            if not lanes:
                continue
            
            total_density = 0.0
            total_queue = 0.0
            
            for lane in lanes:
                veh_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane)
                halting = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                length = self.ts.sumo.lane.getLength(lane)
                
                total_density += min(1.0, veh_count / (length / 7.5))
                total_queue += min(1.0, halting / (length / 5.0))
            
            standardized_density[movement_idx] = total_density / len(lanes)
            standardized_queue[movement_idx] = total_queue / len(lanes)

        for ped_idx in range(4):
            edges = self.pedestrian_edges[ped_idx]
            ped_count = 0
            
            for edge in edges:
                try:
                    ped_count += len(self.ts.sumo.edge.getLastStepPersonIDs(edge))
                except Exception:
                    pass
            
            standardized_peds[ped_idx] = min(1.0, ped_count / 10.0)

        MAX_PHASES = 8
        phase_id = [0] * MAX_PHASES
        
        if self.ts.green_phase < MAX_PHASES:
            phase_id[self.ts.green_phase] = 1
            
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        obs = np.concatenate([
            phase_id,
            min_green,
            standardized_density,
            standardized_queue,
            standardized_peds
        ])

        return obs.astype(np.float32)

    def observation_space(self):
        """Returns a fixed-size Box space."""
        self._setup()
        
        MAX_PHASES = 8
        total_len = MAX_PHASES + 1 + 8 + 8 + 4
        
        return Box(low=0.0, high=1.0, shape=(total_len,), dtype=np.float32)



class NemaStandardizedObservation(ObservationFunction):
    def __init__(self, traffic_signal):
        super().__init__(traffic_signal)
        self.ts = traffic_signal
        self._setup_done = False
        
        # This will hold lists of lane IDs mapped to our 8 standard indices
        # Indices: 0:NB_SR, 1:NB_L, 2:EB_SR, 3:EB_L, 4:SB_SR, 5:SB_L, 6:WB_SR, 7:WB_L
        self.movement_lanes = {i: [] for i in range(8)}

    def _get_bearing(self, lane_id):
        """Calculates the compass bearing of an incoming edge pointing toward the junction."""
        shape = self.ts.sumo.lane.getShape(lane_id)
        # Get the second to last and last point of the edge
        x1, y1 = shape[-2][:2]
        x2, y2 = shape[-1][:2]
        
        # Calculate angle and convert to compass bearing (0 is North, 90 is East)
        angle_rad = math.atan2(x2 - x1, y2 - y1)
        bearing = (math.degrees(angle_rad) + 360) % 360
        return bearing

    def _determine_approach_direction(self, bearing):
        """Maps a compass bearing to an Approach Direction (N, E, S, W)."""
        # Note: If an edge is pointing South (bearing ~180), the approach is FROM the North.
        if 135 <= bearing < 225:
            return "N" # Northbound approach (traffic moving South)
        elif 225 <= bearing < 315:
            return "E" # Eastbound approach (traffic moving West)
        elif 315 <= bearing <= 360 or 0 <= bearing < 45:
            return "S" # Southbound approach (traffic moving North)
        elif 45 <= bearing < 135:
            return "W" # Westbound approach (traffic moving East)

    def _setup(self):
        """Maps arbitrary SUMO lanes to the 8 standard movements."""
        if self._setup_done:
            return

        incoming_lanes = self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)
        # Remove duplicates (SUMO sometimes lists lanes multiple times for different phases)
        incoming_lanes = list(dict.fromkeys(incoming_lanes))

        for lane in incoming_lanes:
            edge_id = self.ts.sumo.lane.getEdgeID(lane)
            bearing = self._get_bearing(lane)
            approach_dir = self._determine_approach_direction(bearing)

            # Look at where this lane goes to determine if it's a left turn or straight/right
            links = self.ts.sumo.lane.getLinks(lane)
            is_left_turn = False
            
            for link in links:
                # link[2] contains the direction string ('s'=straight, 'l'=left, 'L'=partially left, 'r'=right)
                if link[2] in ['l', 'L']:
                    is_left_turn = True
                    break

            # Map to our 8 indices based on Direction and Turn type
            if approach_dir == "N":
                idx = 1 if is_left_turn else 0
            elif approach_dir == "E":
                idx = 3 if is_left_turn else 2
            elif approach_dir == "S":
                idx = 5 if is_left_turn else 4
            elif approach_dir == "W":
                idx = 7 if is_left_turn else 6

            self.movement_lanes[idx].append(lane)

        self._setup_done = True

    def __call__(self):
        """Fetches the standardized 8-movement observation state."""
        self._setup()

        # Initialize fixed-size arrays for our metrics
        standardized_density = np.zeros(8, dtype=np.float32)
        standardized_queue = np.zeros(8, dtype=np.float32)

        for movement_idx in range(8):
            lanes = self.movement_lanes[movement_idx]
            
            if not lanes:
                # Missing leg or missing turn lane; leave as 0.0
                continue
            
            total_density = 0.0
            total_queue = 0.0
            
            for lane in lanes:
                # Add domain randomization noise here if you want to bridge sim-to-real!
                veh_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane)
                halting = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                length = self.ts.sumo.lane.getLength(lane)
                
                # Normalize metrics
                total_density += min(1.0, veh_count / (length / 7.5))
                total_queue += min(1.0, halting / (length / 5.0))
            
            # Average the metrics if multiple lanes exist for this single movement
            standardized_density[movement_idx] = total_density / len(lanes)
            standardized_queue[movement_idx] = total_queue / len(lanes)

        # Force the phase array to always be length 8
        MAX_PHASES = 8
        phase_id = [0] * MAX_PHASES
        
        # Only set the current phase to 1 if it fits within our max bounds
        # (This prevents index errors if SUMO throws a weird phase at us)
        if self.ts.green_phase < MAX_PHASES:
            phase_id[self.ts.green_phase] = 1
            
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        # Combine into a flat, predictable, 1D array
        obs = np.concatenate([
            phase_id,
            min_green,
            standardized_density,
            standardized_queue
        ])

        return obs.astype(np.float32)

    def observation_space(self):
        """Returns a fixed-size Box space."""
        self._setup()
        
        # 8 (fixed max phases) + 1 (min_green) + 8 (density) + 8 (queue)
        MAX_PHASES = 8
        total_len = MAX_PHASES + 1 + 8 + 8
        
        from gymnasium.spaces import Box
        return Box(low=0.0, high=1.0, shape=(total_len,), dtype=np.float32)
    

'''
REWARD FUNCTIONS
'''

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
            'avg_veh': 0.0,
            'avg_ped': 0.0,
            'max_lane_wait': 0.0,
            'p95_ped': 0.0,
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
    max_lane_wait = max(lane_wait_times) if lane_wait_times else 0.0

    '''pedestrian metrics'''
    ped_wait_times = []
    for edge in traffic_signal.pedestrian_edges:
        p_ids = traffic_signal.sumo.edge.getLastStepPersonIDs(edge)
        for p_id in p_ids:
            ped_wait_times.append(traffic_signal.sumo.person.getWaitingTime(p_id))
    
    pedestrian_delay = sum(ped_wait_times)
    p95_ped_wait = np.percentile(ped_wait_times, 95) if ped_wait_times else 0.0

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
    
    total_edges = len(traffic_signal.incoming_edges) + len(traffic_signal.outgoing_edges)
    current_pressure = abs(in_count - out_count) / max(1, total_edges)

    '''calculate and clip deltas'''
    v_delay_delta = np.clip(traffic_signal.prev_stats['avg_veh'] - avg_veh, -5.0, 5.0)
    p_delay_delta = np.clip(traffic_signal.prev_stats['avg_ped'] - avg_ped, -5.0, 5.0)
    
    max_lane_delta = np.clip(traffic_signal.prev_stats['max_lane_wait'] - max_lane_wait, -10.0, 10.0)
    p95_delta = np.clip(traffic_signal.prev_stats['p95_ped'] - p95_ped_wait, -10.0, 10.0)
    
    equity_delta = current_equity_idx - traffic_signal.prev_stats['equity_idx']
    pressure_delta = np.clip(traffic_signal.prev_stats['pressure'] - current_pressure, -2.0, 2.0)

    '''switching penalty'''
    current_phase = traffic_signal.green_phase
    switching_penalty = 1.0 if (traffic_signal.prev_stats['phase'] is not None and 
                               current_phase != traffic_signal.prev_stats['phase']) else 0.0

    '''Reward Calculation'''
    w_veh = 1.0
    w_ped = 1.0
    w_fair = 0.5
    w_equity = 3.0
    w_switch = 0.5
    w_pressure = 1.5

    reward = (
        (w_veh * v_delay_delta) + 
        (w_ped * p_delay_delta) + 
        (w_fair * max_lane_delta) + 
        (w_fair * p95_delta) +
        (w_equity * equity_delta) +
        (w_pressure * pressure_delta) -
        (w_switch * switching_penalty)
    )

    reward = np.clip(reward, -10.0, 10.0)

    traffic_signal.prev_stats = {
        'avg_veh': avg_veh,
        'avg_ped': avg_ped,
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
            'avg_veh': 0.0,
            'max_lane_wait': 0,
            'phase': None,
            'pressure': 0.0
        }


    lane_wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    vehicle_delay = sum(lane_wait_times)
    max_lane_wait = max(lane_wait_times) if lane_wait_times else 0

    vehicle_waiting_count = traffic_signal.get_total_queued()
    avg_veh = vehicle_delay / vehicle_waiting_count if vehicle_waiting_count > 0 else 0.0

    in_count = sum(traffic_signal.sumo.edge.getLastStepVehicleNumber(e) for e in traffic_signal.incoming_edges)
    out_count = sum(traffic_signal.sumo.edge.getLastStepVehicleNumber(e) for e in traffic_signal.outgoing_edges)
    

    total_edges = len(traffic_signal.incoming_edges) + len(traffic_signal.outgoing_edges)
    current_pressure = abs(in_count - out_count) / max(1, total_edges)

    v_delay_delta = np.clip(traffic_signal.prev_stats['avg_veh'] - avg_veh, -5.0, 5.0)
    max_lane_delta = np.clip(traffic_signal.prev_stats['max_lane_wait'] - max_lane_wait, -10.0, 10.0)
    pressure_delta = np.clip(traffic_signal.prev_stats['pressure'] - current_pressure, -2.0, 2.0)


    current_phase = traffic_signal.green_phase
    switching_penalty = 1.0 if (traffic_signal.prev_stats['phase'] is not None and 
                               current_phase != traffic_signal.prev_stats['phase']) else 0.0

    w_veh = 1.0
    w_max_wait = 0.5
    w_switch = 0.5
    w_pressure = 1.5

    reward = (
        (w_veh * v_delay_delta) + 
        (w_max_wait * max_lane_delta) + 
        (w_pressure * pressure_delta) -
        (w_switch * switching_penalty)
    )

    traffic_signal.prev_stats = {
        'avg_veh': avg_veh,
        'max_lane_wait': max_lane_wait,
        'phase': current_phase,
        'pressure': current_pressure
    }

    return reward


'''
MISCELLANEOUS
'''

def parse_true_metrics(tripinfo_path):
    """Parses SUMO's native tripinfo.xml for 100% accurate wait times."""
    try:
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Warning: {tripinfo_path} not found. Returning zeros.")
        return {'veh_count': 0, 'veh_avg': 0.0, 'veh_p95': 0.0,
                'ped_count': 0, 'ped_avg': 0.0, 'ped_p95': 0.0, 'cross_modal_gap': 0.0}

    veh_waits = []
    ped_waits = []

    for trip in root.findall('tripinfo'):
        wait = float(trip.get('waitingTime', 0))
        veh_waits.append(wait)

    for person in root.findall('personinfo'):
        p_wait = 0.0
        for walk in person.findall('walk'):
            p_wait += float(walk.get('timeLoss', 0))
        ped_waits.append(p_wait)

    v_95 = np.percentile(veh_waits, 95) if veh_waits else 0.0
    v_avg = np.mean(veh_waits) if veh_waits else 0.0
    v_count = len(veh_waits)

    p_95 = np.percentile(ped_waits, 95) if ped_waits else 0.0
    p_avg = np.mean(ped_waits) if ped_waits else 0.0
    p_count = len(ped_waits)

    return {
        'veh_count': v_count, 'veh_avg': v_avg, 'veh_p95': v_95,
        'ped_count': p_count, 'ped_avg': p_avg, 'ped_p95': p_95,
        'cross_modal_gap': abs(v_avg - p_avg)
    }

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