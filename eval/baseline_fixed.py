import os
import sys
import traci
import csv

def get_intersection_metrics(tl_id):
    """
    helper function to extract wait times and counts for an intersection.
    """

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


def run_fixed_timer():
    """
    runs the SUMO simulation using the default fixed-timer traffic lights
    acts as your absolute baseline control group
    """

    sumoCmd = ["sumo", "-c", "simulation/sim.sumocfg"]
    traci.start(sumoCmd)

    step = 0

    with open('fixed_timer_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            'step',
            'vehicle_total_stopped',
            'vehicle_total_waiting_time',
            'vehicle_average_waiting_time',
            'pedestrian_total_stopped',
            'pedestrian_total_waiting_time',
            'pedestrian_average_waiting_time'
        ])

        while step < 3600:

            traci.simulationStep()

            if step % 5 == 0:

                tl_ids = traci.trafficlight.getIDList()

                net_veh_count = 0
                net_veh_time = 0
                net_ped_count = 0
                net_ped_time = 0

                for tl_id in tl_ids:
                    v_count, v_time, p_count, p_time = get_intersection_metrics(tl_id)

                    net_veh_count += v_count
                    net_veh_time += v_time
                    net_ped_count += p_count
                    net_ped_time += p_time

                # Compute averages safely
                veh_avg_wait = net_veh_time / net_veh_count if net_veh_count > 0 else 0
                ped_avg_wait = net_ped_time / net_ped_count if net_ped_count > 0 else 0

                writer.writerow([
                    step,
                    net_veh_count,
                    net_veh_time,
                    veh_avg_wait,
                    net_ped_count,
                    net_ped_time,
                    ped_avg_wait
                ])

                file.flush()

            step += 1

    traci.close()
    print("Fixed-Timer Baseline Complete.")


if __name__ == "__main__":
    run_fixed_timer()
