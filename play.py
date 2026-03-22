"""
play.py - Load the trained DQN model and play Atari Assault
=============================================================
Loads the best trained model and runs the agent with visual display.

Usage:
    python play.py
    python play.py --episodes 10
"""

import argparse
import os
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN


def play(model_path, num_episodes=5):
    """Load the model and play the game with visual rendering."""

    print(f"\nLoading model from: {model_path}")
    model = DQN.load(model_path)

    # Create environment with human rendering (opens a game window)
    env = gym.make("ALE/Assault-v5", render_mode="human", full_action_space=False)
    print(f"\nPlaying {num_episodes} episodes...\n")

    all_rewards = []
    all_lengths = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Greedy policy: deterministic=True ensures agent picks
            # the action with the highest Q-value (no random exploration)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.1f}  |  Length = {episode_length}")

    env.close()

    # Print summary
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_length = sum(all_lengths) / len(all_lengths)

    print("\n" + "=" * 60)
    print("  GAMEPLAY SUMMARY")
    print("=" * 60)
    print(f"  Episodes Played: {num_episodes}")
    print(f"  Average Reward:  {avg_reward:.1f}")
    print(f"  Best Reward:     {max(all_rewards):.1f}")
    print(f"  Worst Reward:    {min(all_rewards):.1f}")
    print(f"  Average Length:  {avg_length:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Atari Assault with trained DQN agent")
    parser.add_argument("--model_path", type=str, default=os.path.join("models", "dqn_model.zip"),
                        help="Path to model .zip file (default: models/dqn_model.zip)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found at: {args.model_path}")
        print("Make sure dqn_model.zip is in the models/ folder.")
    else:
        play(args.model_path, args.episodes)