import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from utils import NemaStandardizedObservation, SaveVecNormalizeCallback, fair_wait_time_reward

save_dir = "./models17"
custom_checkpoint_callback = SaveVecNormalizeCallback(
    save_freq=50000, 
    save_path=save_dir,
    name_prefix="ppo_model",
    verbose=1
)

def main():
    # creating multi-agent environment
    env = sumo_rl.parallel_env(
        net_file='simulation/grid-network.net.xml',
        route_file='simulation/vehs.rou.xml,simulation/peds.rou.xml',
        use_gui=False,
        num_seconds=3600,
        reward_fn=fair_wait_time_reward,
        observation_class=NemaStandardizedObservation,
        sumo_seed='random',
    )

    env.unwrapped.render_mode = None    
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=4, base_class='stable_baselines3')

    env = VecMonitor(env)

    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # initial proximal policy optimization (PPO) model
    alpha = 0.0003
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=3,
        learning_rate=alpha,
        gamma=0.95,
        tensorboard_log="./ppo_tensorboard/",
        ent_coef=0.05,
        batch_size=256 
    )

    model.learn(total_timesteps=10000000, callback=custom_checkpoint_callback)

    model.save("traffic_model_generalized")
    env.save("vec_normalize.pkl")

    env.close()

if __name__ == "__main__":
    main()