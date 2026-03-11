import sumo_rl
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import NeighborObservation, fair_wait_time_reward
import os

# deque to keep track of previous traffic states to prevent rapid switching
# dictionary of deques for each intersection
BUFFER_SIZE = 10

NEIGHBORS_DICT = {
    'B1': ['B2', 'C1'],
    'B2': ['B1', 'C2'],
    'C1': ['B1', 'C2'],
    'C2': ['C1', 'B2'],
}

# saving model mid-training

save_dir = "./models1"
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=save_dir,
    name_prefix="ppo_model"
)


def main():
    '''
    1. initializes agent environment
    2. vectorizes to compatable with stable_baselines3
    3. PPO model initialization and training
    4. saves model
    '''

    # creating multi-agent environment
    env = sumo_rl.parallel_env(
        net_file='grid-network.net.xml',
        route_file='vehs.rou.xml,peds.rou.xml',
        out_csv_name='results',
        use_gui=False,
        num_seconds=20000,
        reward_fn=fair_wait_time_reward,
        observation_class=NeighborObservation,
        sumo_seed=42
    )

    # used to make compatible
    env.unwrapped.render_mode = None

    # padding to observation space necessary,
    # because not all traffic signals will have same number of neighbors
    env = ss.pad_observations_v0(env)

    # vectorization of pettingzoo to stable baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # initial proximal policy optimization (PPO) model
    alpha = 0.0003
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=3,
        learning_rate=alpha,
        gamma=0.95,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=10000000, callback=checkpoint_callback)

    model.save("traffic_model")

    env.close()


if __name__ == "__main__":
    main()