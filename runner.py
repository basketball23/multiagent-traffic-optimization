import os
import sys
import time
import traci

'''
from gymenv import SumoIntersectionEnv

from stable_baselines3 import DQN

env = SumoIntersectionEnv()

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)

model.learn(total_timesteps=100000)
'''

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: 'SUMO_HOME' env variable is not declared")


def run_simulation():
    step = 0

    while traci.simulation.getMinExpectedNumber() > 0:

        # runs one tick of the simulation
        traci.simulationStep()

        # collects data every 5 steps
        if step % 5 == 0:

            # counts cars waiting at a red light
            vehicles = traci.vehicle.getIDList()
            waiting_cars = 0

            for v in vehicles:
                if traci.vehicle.getWaitingTime(v) > 0:
                    waiting_cars += 1
            
            print(f"Time {step}: Waiting cars = {waiting_cars}")


            # fetches traffic light states
            tl_ids = traci.trafficlight.getIDList()

            if len(tl_ids) > 0:

                for i in range(len(tl_ids)):

                    target_light = tl_ids[i]
                    current_phase = traci.trafficlight.getPhase(target_light)

                    print(f"Time {step}: Traffic Light '{target_light}' is in phase {current_phase}")


        step += 1


        # add custom logic
    
    traci.close()
    sys.stdout.flush()



if __name__ == "__main__":
    sumoCmd = ["sumo-gui", "-c", "sim.sumocfg"]

    traci.start(sumoCmd)
    print("Successfully connected to SUMO")

    run_simulation()