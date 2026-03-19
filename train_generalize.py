import os
import sys
import random
import subprocess
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from utils import NemaStandardizedObservation, SaveVecNormalizeCallback, fair_wait_time_reward

# --- SUPERSUIT MULTIPROCESSING BUG FIX ---
import supersuit.vector.vector_constructors as vc

def patched_MakeCPUAsyncConstructor(num_cpus):
    def constructor(*args, **kwargs):
        from supersuit.vector.multiproc_vec import SubprocVecEnv
        env_fns = args[0]
        dummy_env = env_fns[0]()
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        dummy_env.close()
        return SubprocVecEnv(env_fns, obs_space, act_space, **kwargs)
    return constructor

vc.MakeCPUAsyncConstructor = patched_MakeCPUAsyncConstructor
# -----------------------------------------

if 'SUMO_HOME' not in os.environ:
    sys.exit("Error: Please declare environment variable 'SUMO_HOME'")

def generate_random_traffic(net_file, route_file, sim_seconds=3600):
    """
    Dynamically generates a SUMO route file with a random traffic volume.
    """
    randomTrips_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    period = round(random.uniform(1.0, 4.0), 2)
    print(f"    -> Generating new traffic volume. Arrival period: {period}s")

    cmd = [
        sys.executable, randomTrips_path,
        "-n", net_file,
        "-r", route_file,
        "-e", str(sim_seconds),
        "-p", str(period),
        "--fringe-factor", "10"
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    save_dir = "./models_curriculum"
    model_path = os.path.join(save_dir, "traffic_model_generalized")
    vec_norm_path = os.path.join(save_dir, "vec_normalize.pkl")
    
    os.makedirs(save_dir, exist_ok=True)

    custom_checkpoint_callback = SaveVecNormalizeCallback(
        save_freq=50000, 
        save_path=save_dir,
        name_prefix="ppo_model",
        verbose=0
    )

    curriculum = [
        {'net': 'simulation/grid-network.net.xml', 'route_name': 'simulation/dynamic_grid.rou.xml'},
        {'net': 'simulation/t-junction.net.xml', 'route_name': 'simulation/dynamic_t.rou.xml'},
        {'net': 'simulation/arterial.net.xml', 'route_name': 'simulation/dynamic_art.rou.xml'}
    ]

    total_steps_per_map = 2_000_000
    volume_swaps_per_map = 4 
    steps_per_volume = total_steps_per_map // volume_swaps_per_map # 500,000 steps

    sim_seconds = 3600

    for stage_idx, stage in enumerate(curriculum):
        print(f"\n{'='*50}")
        print(f"STAGE {stage_idx + 1}: Training on Topology -> {stage['net']}")
        print(f"{'='*50}")

        for volume_idx in range(volume_swaps_per_map):
            print(f"\n  [Volume Swap {volume_idx + 1}/{volume_swaps_per_map}]")
            
            # 2. Generate fresh, randomized traffic for this specific training chunk
            generate_random_traffic(stage['net'], stage['route_name'], sim_seconds)
            
            # 3. Initialize environment
            env = sumo_rl.parallel_env(
                net_file=stage['net'],
                route_file=stage['route_name'],
                use_gui=False,
                num_seconds=sim_seconds,
                reward_fn=fair_wait_time_reward,
                observation_class=NemaStandardizedObservation,
                sumo_seed='random',
            )
            
            env.unwrapped.render_mode = None
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=4, base_class='stable_baselines3')
            env = VecMonitor(env)

            # 4. Handle Persistent Normalization Stats
            if os.path.exists(vec_norm_path):
                env = VecNormalize.load(vec_norm_path, env)
                env.training = True 
                env.norm_reward = False
            else:
                env = VecNormalize(env, norm_obs=True, norm_reward=False)

            # 5. Handle Persistent Model Weights
            if os.path.exists(model_path + ".zip"):
                model = PPO.load(model_path, env=env)
                
                # Optional: Decay learning rate slightly as we get deeper into the curriculum
                if stage_idx > 0:
                    model.learning_rate = 0.0001 
            else:
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

            # 6. Train for this volume chunk
            model.learn(total_timesteps=steps_per_volume, reset_num_timesteps=False, callback=custom_checkpoint_callback)

            # 7. Save and safely close before the next volume swap
            model.save(model_path)
            env.save(vec_norm_path)
            env.close()

if __name__ == "__main__":
    main()