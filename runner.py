import os
import sys
import traci

import sumo_rl

# check env variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Error: 'SUMO_HOME' env variable is not declared")


# create pettingzoo parallel environment for agents

env = sumo_rl.parallel_env(
    net_file='test-network.net.xml',
    route_file='vehicles.rou.xml',
    use_gui=True,
    num_seconds=3600,
    #reward_fn=fair_wait_time_reward
)


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