import os
import sys
import traci
import csv

MIN_GREEN_TIME = 15      
QUEUE_THRESHOLD = 3      

def get_intersection_metrics(tl_id):
    """Helper function to extract wait times and counts for an intersection."""
    veh_wait_count = 0
    veh_wait_time = 0
    ped_wait_count = 0
    ped_wait_time = 0

    lanes = set(traci.trafficlight.getControlledLanes(tl_id))
    
    for lane in lanes:
        edge_id = traci.lane.getEdgeID(lane)
        if "crosswalk" in edge_id or "walking" in edge_id:
            ped_wait_count += traci.edge.getLastStepHaltingNumber(edge_id)
            ped_wait_time += traci.edge.getWaitingTime(edge_id)
        else:
            veh_wait_count += traci.lane.getLastStepHaltingNumber(lane)
            veh_wait_time += traci.lane.getWaitingTime(lane)
            
    return veh_wait_count, veh_wait_time, ped_wait_count, ped_wait_time

def run_multi_rule_based():
    # Set to "sumo" instead of "sumo-gui" for much faster data generation
    sumoCmd = ["sumo", "-c", "simulation/sim.sumocfg"]
    traci.start(sumoCmd)
    
    print("Running Rule-Based Control and logging data...")
    
    step = 0
    phase_timers = {} 
    
    with open('rule_based_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['step', 'vehicle_total_stopped', 'vehicle_total_waiting_time', 'pedestrian_total_stopped', 'pedestrian_total_waiting_time'])
        
        while step < 3600:
            traci.simulationStep()
            tl_ids = traci.trafficlight.getIDList()
            
            for target_light in tl_ids:
                if target_light not in phase_timers:
                    phase_timers[target_light] = 0
                    
                phase_timers[target_light] += 1
                current_phase = traci.trafficlight.getPhase(target_light)
                
                if current_phase == 0 or current_phase == 2:
                    if phase_timers[target_light] >= MIN_GREEN_TIME:
                        waiting_entities = 0
                        lanes = traci.trafficlight.getControlledLanes(target_light)
                        
                        for lane in lanes:
                            if current_phase == 0 and ("E2" in lane or "W2" in lane): 
                                waiting_entities += traci.lane.getLastStepHaltingNumber(lane)
                            elif current_phase == 2 and ("N2" in lane or "S2" in lane):
                                waiting_entities += traci.lane.getLastStepHaltingNumber(lane)
                        
                        if waiting_entities >= QUEUE_THRESHOLD:
                            next_phase = (current_phase + 1) % 4
                            traci.trafficlight.setPhase(target_light, next_phase)
                            phase_timers[target_light] = 0 
                
                elif current_phase == 1 or current_phase == 3:
                    phase_timers[target_light] = 0


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
                    
                writer.writerow([step, net_veh_count, net_veh_time, net_ped_count, net_ped_time])
                
                file.flush()

            step += 1

    traci.close()
    print("Rule-Based Simulation Complete. Data saved to rule_based_data.csv.")

if __name__ == "__main__":
    run_multi_rule_based()