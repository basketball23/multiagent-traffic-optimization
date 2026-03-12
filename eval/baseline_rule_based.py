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
        if "_w" in edge_id:
            pedestrian_ids = traci.edge.getLastStepPersonIDs(edge_id)
            ped_wait_count += len(pedestrian_ids)

            for p_id in pedestrian_ids:
                ped_wait_time += traci.person.getWaitingTime(p_id)
        else:
            veh_wait_count += traci.lane.getLastStepHaltingNumber(lane)
            veh_wait_time += traci.lane.getWaitingTime(lane)
            
    return veh_wait_count, veh_wait_time, ped_wait_count, ped_wait_time

def run_multi_rule_based():
    sumoCmd = ["sumo", "-c", "simulation/sim.sumocfg"]
    traci.start(sumoCmd)
    
    print("Running Dynamic Rule-Based Control and logging data...")
    
    step = 0
    phase_timers = {} 
    
    with open('rule_based_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['step', 'vehicle_total_stopped', 'vehicle_total_waiting_time', 'pedestrian_total_stopped', 'pedestrian_total_waiting_time'])
        
        while step < 3600:
            traci.simulationStep()
            tl_ids = traci.trafficlight.getIDList()
            
            # --- 1. Dynamic Traffic Light Logic ---
            for target_light in tl_ids:
                if target_light not in phase_timers:
                    phase_timers[target_light] = 0
                    # FIX: Hijack the timer so SUMO doesn't auto-change it
                    traci.trafficlight.setPhaseDuration(target_light, 1000)
                    
                phase_timers[target_light] += 1
                current_phase = traci.trafficlight.getPhase(target_light)
                
                # Even phases (0, 2) are Green
                if current_phase % 2 == 0: 
                    if phase_timers[target_light] >= MIN_GREEN_TIME:
                        waiting_entities = 0
                        
                        # FIX: Dynamic Queue Checking (looks for any red light)
                        state_string = traci.trafficlight.getRedYellowGreenState(target_light)
                        lanes = traci.trafficlight.getControlledLanes(target_light)
                        
                        for i, lane in enumerate(lanes):
                            if state_string[i].lower() == 'r':
                                waiting_entities += traci.lane.getLastStepHaltingNumber(lane)
                        
                        if waiting_entities >= QUEUE_THRESHOLD:
                            next_phase = (current_phase + 1) % 4
                            traci.trafficlight.setPhase(target_light, next_phase)
                            phase_timers[target_light] = 0 
                
                else:
                    if phase_timers[target_light] >= 4: # Run yellow for exactly 4 seconds
                        next_phase = (current_phase + 1) % 4
                        traci.trafficlight.setPhase(target_light, next_phase)
                        # Lock the new green light from auto-changing
                        traci.trafficlight.setPhaseDuration(target_light, 1000)
                        phase_timers[target_light] = 0

            if step % 5 == 0:
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