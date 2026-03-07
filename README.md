# About:
- This project attempts to use Multi-Agent Reinforcement Learning (MARL) to minimize joint pedestrian-vehicle waiting times

## Pipeline will go:
1. SUMO runs simulation step
2. TraCI fetches information/data every x seconds using built-in functions
    - Takes data from each intersection
    - Formats for multiple agents
3. Data is fed into multi-agent system (PettingZoo, stable-baselines3, sumo-rl)
4. RL loop:
    - Error calculation
5. Repeat
