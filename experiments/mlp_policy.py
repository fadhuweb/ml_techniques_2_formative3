
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


# STEP 1: Create the environment
env = gym.make("ALE/Assault-v5", obs_type="ram")
env = Monitor(env)


# STEP 2: Define the DQN agent with BASELINE hyperparameters
model = DQN(
    policy="MlpPolicy",              
    env=env,
    learning_rate=0.0001,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_fraction=0.1,
    buffer_size=50000,
    learning_starts=1000,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    verbose=1,
    seed=42,
)


# STEP 3: Train for 500,000 timesteps
print("=" * 60)
print("  BASELINE TRAINING: MLP Policy on Assault (RAM obs)")
print("  Timesteps: 500,000")
print("=" * 60)

model.learn(total_timesteps=500000, log_interval=10)


# STEP 4: Save the baseline model
model.save("models/baseline_mlp_model")
print("\nModel saved to models/baseline_mlp_model.zip")


# STEP 5: Evaluate over 10 episodes and print results
print("\n" + "=" * 60)
print("  BASELINE EVALUATION (10 episodes)")
print("=" * 60)

eval_env = gym.make("ALE/Assault-v5", obs_type="ram")
rewards = []
lengths = []

for episode in range(10):
    obs, info = eval_env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

    rewards.append(episode_reward)
    lengths.append(episode_length)
    print(f"  Episode {episode + 1:2d}: Reward = {episode_reward:8.1f}  |  Length = {episode_length}")

avg_reward = sum(rewards) / len(rewards)
avg_length = sum(lengths) / len(lengths)
best_reward = max(rewards)
worst_reward = min(rewards)

print("\n" + "-" * 60)
print(f"  Average Reward:  {avg_reward:.1f}")
print(f"  Best Reward:     {best_reward:.1f}")
print(f"  Worst Reward:    {worst_reward:.1f}")
print(f"  Average Length:  {avg_length:.1f}")
print("-" * 60)



print(f"  Policy:          MlpPolicy")
print(f"  Environment:     ALE/Assault-v5 (obs_type=ram)")
print(f"  Timesteps:       500,000")
print(f"  Avg Reward:      {avg_reward:.1f}")
print(f"  Best Reward:     {best_reward:.1f}")
print(f"  Worst Reward:    {worst_reward:.1f}")
print(f"  Avg Ep Length:   {avg_length:.1f}")


env.close()
eval_env.close()