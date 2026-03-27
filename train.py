import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from utils import NemaPedestrianStandardizedObservation, NemaStandardizedObservation, SaveVecNormalizeCallback, fair_wait_time_reward, vehicle_baseline_reward

def main():
    # creating multi-agent environment
    env = sumo_rl.parallel_env(
        net_file='simulation/grid-network-nema.net.xml',
        route_file='simulation/vehs-nema.rou.xml,simulation/peds-nema.rou.xml',
        use_gui=False,
        num_seconds=3600,
        reward_fn=fair_wait_time_reward,
        observation_class=NemaPedestrianStandardizedObservation,
        sumo_seed='random',
    )

    # env processing
    env.unwrapped.render_mode = None    
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=1, base_class='stable_baselines3')

    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # saving
    save_dir = "./models20"
    adjusted_save_freq = max(50000 // env.num_envs, 1)
    
    custom_checkpoint_callback = SaveVecNormalizeCallback(
        save_freq=adjusted_save_freq, 
        save_path=save_dir,
        name_prefix="ppo_model",
        verbose=1
    )

    # initialize ppo model
    alpha = 0.0003
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=alpha,
        gamma=0.95,
        tensorboard_log="./ppo_tensorboard/",
        ent_coef=0.05,
        batch_size=256 
    )

    # run training
    model.learn(total_timesteps=10000000, callback=custom_checkpoint_callback)

    model.save("traffic_model")
    env.save("vec_normalize.pkl")

    env.close()

if __name__ == "__main__":
    main()