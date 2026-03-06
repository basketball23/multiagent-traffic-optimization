import os
import sys
import time

from gymenv import SumoIntersectionEnv

from stable_baselines3 import DQN

env = SumoIntersectionEnv()

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)

model.learn(total_timesteps=100000)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("'SUMO_HOME' is not declared")

import traci

def run_simulation():
    # checks if there are still vehicles left
    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()
        time.sleep(0.1)

        # add custom logic
    
    traci.close()
    sys.stdout.flush()



if __name__ == "__main__":
    sumoBinary = "sumo-gui"
    sumoCmd = [sumoBinary, "-c", "simulation.sumocfg.txt"]

    traci.start(sumoCmd)

    run_simulation()