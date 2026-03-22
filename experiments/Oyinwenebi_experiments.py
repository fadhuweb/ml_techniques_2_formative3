import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import os

# Create environment
def make_env():
    env = gym.make("ALE/Assault-v5")
    env = AtariWrapper(env)
    return env

# List of experiments (10 configs)
experiments = [
    {"lr": 1e-3, "gamma": 0.95, "batch": 32, "eps_start":1.0, "eps_end":0.1, "eps_frac":0.1},
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start":1.0, "eps_end":0.05, "eps_frac":0.1},
    {"lr": 5e-5, "gamma": 0.99, "batch": 64, "eps_start":1.0, "eps_end":0.01, "eps_frac":0.2},
    {"lr": 1e-4, "gamma": 0.90, "batch": 32, "eps_start":1.0, "eps_end":0.1, "eps_frac":0.1},
    {"lr": 1e-4, "gamma": 0.99, "batch": 128, "eps_start":1.0, "eps_end":0.05, "eps_frac":0.1},
    {"lr": 1e-3, "gamma": 0.99, "batch": 32, "eps_start":1.0, "eps_end":0.05, "eps_frac":0.1},
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start":0.8, "eps_end":0.05, "eps_frac":0.1},
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start":1.0, "eps_end":0.01, "eps_frac":0.05},
    {"lr": 5e-5, "gamma": 0.98, "batch": 64, "eps_start":1.0, "eps_end":0.05, "eps_frac":0.2},
    {"lr": 1e-4, "gamma": 0.995, "batch": 32, "eps_start":1.0, "eps_end":0.05, "eps_frac":0.1},
]

# Loop through experiments
for i, exp in enumerate(experiments):
    print(f"\n🚀 Running Experiment {i+1}")

    env = DummyVecEnv([make_env])

    log_dir = f"./logs/exp_{i+1}/"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "csv"])

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        batch_size=exp["batch"],
        buffer_size=100000,
        learning_starts=10000,
        exploration_initial_eps=exp["eps_start"],
        exploration_final_eps=exp["eps_end"],
        exploration_fraction=exp["eps_frac"],
        verbose=1,
    )

    model.set_logger(logger)

    model.learn(total_timesteps=50000)  # keep small for faster experiments

    model.save(f"dqn_model_exp_{i+1}")

print("✅ All experiments completed!")