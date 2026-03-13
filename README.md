# About:
- This project attempts to use Multi-Agent Reinforcement Learning (MARL) with centralized training and decentralized execution to minimize joint pedestrian-vehicle waiting times
- Employing Parameter-Shared MARL with Decentralized Cooperative Agents to Achieve Fair and Efficient Wait Times for Pedestrians and Vehicles

## Pipeline will go:
1. SUMO runs simulation step
2. TraCI fetches information/data every x seconds using built-in functions. Each intersection observes:
    - Default data (pressure, density, traffic light phases, min time since change)
    - Takes data from neighboring intersections ()
    - Formats for multiple agents
3. Data is fed into multi-agent system (sumo-rl, PettingZoo, stable-baselines3)
4. RL loop:
    - Reward funciton calculation with:
        * Average vehicle delay
        * Average pedestrian delay
        * Equity (abs(avg_veh - avg_ped))
        * P95 pedestrian wait times (prevents starvation)
        * Max lane wait times (prevents starvation)
        * Traffic signal switching penalty
5. Repeat
