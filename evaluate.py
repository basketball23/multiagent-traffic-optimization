import sumo_rl
import supersuit as ss

from stable_baselines3 import PPO
from utils import NeighborObservation, fair_wait_time_reward


def main():
    # creating multi-agent environment
    env = sumo_rl.parallel_env(
        net_file='simulation/grid-network.net.xml',
        route_file='simulation/vehs.rou.xml,simulation/peds.rou.xml',
        out_csv_name='evaluation',
        use_gui=True,
        num_seconds=20000,
        reward_fn=fair_wait_time_reward,
        observation_class=NeighborObservation,
    )

    # used to make compatible
    env.unwrapped.render_mode = None

    env = ss.pad_observations_v0(env)

    # vectorization of pettingzoo to stable baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    model = PPO.load("models1/ppo_model_5600000_steps.zip")

    obs = env.reset()

    dones = [False]
    while not all(dones):
        # deterministic to stop taking random actions
        actions, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()