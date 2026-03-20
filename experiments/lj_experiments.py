import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os

# folder for models
os.makedirs("models", exist_ok=True)

# 10 EXPERIMENTS
experiments = [
    # Experiment 1 (Baseline)
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.1},

    # Experiment 2 - Higher learning rate
    {"lr": 5e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.1},

    # Experiment 3 - Lower gamma
    {"lr": 1e-4, "gamma": 0.95, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.1},

    # Experiment 4 - Larger batch
    {"lr": 1e-4, "gamma": 0.99, "batch": 64, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.1},

    # Experiment 5 - Smaller batch
    {"lr": 1e-4, "gamma": 0.99, "batch": 16, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.1},

    # Experiment 6 - More exploration
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.1, "eps_frac": 0.2},

    # Experiment 7 - Less exploration
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start": 0.8, "eps_end": 0.01, "eps_frac": 0.05},

    # Experiment 8 - Faster epsilon decay
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.05},

    # Experiment 9 - Slower epsilon decay
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.3},

    # Experiment 10 - Combined aggressive settings
    {"lr": 5e-4, "gamma": 0.95, "batch": 64, "eps_start": 1.0, "eps_end": 0.1, "eps_frac": 0.2},
]

# RUN EXPERIMENTS
for i, exp in enumerate(experiments):
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT {i+1}")
    print(f"  lr={exp['lr']}, gamma={exp['gamma']}, batch={exp['batch']}, "
          f"eps_start={exp['eps_start']}, eps_end={exp['eps_end']}, eps_frac={exp['eps_frac']}")
    print("=" * 70)

    # environment
    env = gym.make("ALE/Assault-v5")
    env = Monitor(env)

    # model
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        batch_size=exp["batch"],
        exploration_initial_eps=exp["eps_start"],
        exploration_final_eps=exp["eps_end"],
        exploration_fraction=exp["eps_frac"],
        buffer_size=50000,
        learning_starts=1000,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        verbose=0,
        seed=42,
    )

    # Train
    model.learn(total_timesteps=200000, log_interval=10)

    # Save model
    model_path = f"models/dqn_experiment_{i+1}"
    model.save(model_path)
    print(f"Model saved: {model_path}.zip")

    # EVALUATION
    eval_env = gym.make("ALE/Assault-v5")

    rewards = []
    for episode in range(5):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    avg_reward = sum(rewards) / len(rewards)
    print(f"Average Reward: {avg_reward:.2f}")

    env.close()
    eval_env.close()

print("\nALL EXPERIMENTS COMPLETED")