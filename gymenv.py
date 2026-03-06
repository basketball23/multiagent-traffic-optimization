import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

class SumoIntersectionEnv(gym.Env):
    def __init__(self):
        super(SumoIntersectionEnv, self).__init__()

        # agent observation and action spaces
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)

        # [0: keep, 1: N/S G, E/W G, Pedestrian Walk]
        self.action_space = spaces.Discrete(4)

        # hyperparameters for vehicle delay, pedestrian delay, and frequency switching
        self.w1 = 1.0 
        self.w2 = 2.0
        self.w3 = 5.0

        self.current_phase = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # restart SUMO using traci

        initial_state = self._get_state()
        return initial_state, {}
    
    def step(self, action):
        # apply agent action to simulation

        # switching penalty
        penalty = 0
        if action != self.current_phase:
            penalty = 1

            # traci.trafficlight.setPhase("my_intersection", action)

            self.current_phase = action

        
        traci.simulationStep()

        next_state = self._get_state()

        reward = self._calculate_reward(penalty)

        terminated = traci.simulation.getMinExpectedNumber() <= 0
        
        return next_state, reward, terminated, {}
    
    def _get_state(self):
        '''
        Reads data from SUMO using TraCI

        Returns a vector of size 10 with traffic state
        '''

        ## read data from traci about current traffic state



        # placeholder
        return np.zeros(10, dtype=np.float32)
    
    def _calculate_reward(self, switch_penalty):
        
        # read vehicle dealsy from traci
        # placeholders

        d_veh = 10.0
        d_ped = 2.0


        # simple reward function
        reward = -(self.w1 * d_veh + self.w2 * d_ped + self.w3 + switch_penalty)
        return reward
