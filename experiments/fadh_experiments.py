import argparse
import os
from datetime import datetime
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure


def parse_args():
    """Parse command line arguments for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Train DQN Agent on Atari Assault")

    # Experiment tracking
    parser.add_argument("--experiment", type=int, default=0,
                        help="Experiment number (0 = baseline)")
    parser.add_argument("--member_name", type=str, default="Member",
                        help="Your name for documentation")

    # Hyperparameters to tune
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Starting exploration rate (default: 1.0)")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Final exploration rate (default: 0.05)")
    parser.add_argument("--epsilon_decay", type=float, default=0.1,
                        help="Fraction of timesteps over which epsilon decays (default: 0.1)")

    # Training settings
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 10 experiments automatically")

    return parser.parse_args()


def train(args):
    """Train the DQN agent with the given hyperparameters."""

    # =========================================================================
    # STEP 1: Create directories
    # =========================================================================
    experiment_name = f"exp{args.experiment}_lr{args.lr}_g{args.gamma}_bs{args.batch_size}_es{args.epsilon_start}_ee{args.epsilon_end}_ed{args.epsilon_decay}"
    log_dir = os.path.join("..", "logs", experiment_name)
    model_dir = os.path.join("..", "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print(f"  EXPERIMENT {args.experiment}: DQN with CnnPolicy on Assault")
    print("=" * 60)
    print(f"  Member:            {args.member_name}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Gamma:             {args.gamma}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Epsilon Start:     {args.epsilon_start}")
    print(f"  Epsilon End:       {args.epsilon_end}")
    print(f"  Epsilon Decay:     {args.epsilon_decay}")
    print(f"  Total Timesteps:   {args.timesteps}")
    print(f"  Log Directory:     {log_dir}")
    print("=" * 60)

    # =========================================================================
    # STEP 2: Create training and evaluation environments
    # =========================================================================
    train_env = gym.make("ALE/Assault-v5")
    train_env = Monitor(train_env)

    eval_env = gym.make("ALE/Assault-v5")
    eval_env = Monitor(eval_env)

    # =========================================================================
    # STEP 3: Set up logging (TensorBoard + CSV)
    # =========================================================================
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # =========================================================================
    # STEP 4: Define the DQN agent with CnnPolicy
    # =========================================================================
    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay,
        buffer_size=10000,
        learning_starts=1000,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )

    model.set_logger(logger)

    # =========================================================================
    # STEP 5: Set up evaluation callback (evaluates every 50k steps)
    # =========================================================================
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, experiment_name),
        log_path=log_dir,
        eval_freq=50000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # =========================================================================
    # STEP 6: Train the agent
    # =========================================================================
    print("\nStarting training...\n")
    start_time = datetime.now()

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        log_interval=10,
    )

    training_time = datetime.now() - start_time
    print(f"\nTraining completed in {training_time}")

    # =========================================================================
    # STEP 7: Save the BEST model (not the final one)
    # =========================================================================
    # The EvalCallback already saved the best model during training.
    # Copy it to models/dqn_model.zip and experiment-specific path.
    import shutil

    best_model_src = os.path.join(model_dir, experiment_name, "best_model.zip")

    if os.path.exists(best_model_src):
        # Copy best model as the main dqn_model.zip
        shutil.copy(best_model_src, os.path.join(model_dir, "dqn_model.zip"))
        print(f"Best model saved to models/dqn_model.zip")

        # Also copy as experiment-specific model
        shutil.copy(best_model_src, os.path.join(model_dir, f"dqn_model_exp{args.experiment}.zip"))
        print(f"Best model also saved to models/dqn_model_exp{args.experiment}.zip")
    else:
        # Fallback: save final model if best model wasn't saved for some reason
        model.save(os.path.join(model_dir, "dqn_model"))
        model.save(os.path.join(model_dir, f"dqn_model_exp{args.experiment}"))
        print(f"Best model not found, saved final model instead.")

    # Load the best model for evaluation
    best_model_path = os.path.join(model_dir, "dqn_model.zip")
    model = DQN.load(best_model_path)

    # =========================================================================
    # STEP 8: Evaluate over 10 episodes
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"  EXPERIMENT {args.experiment} EVALUATION (10 episodes)")
    print("=" * 60)

    eval_env_final = gym.make("ALE/Assault-v5")
    rewards = []
    lengths = []

    for episode in range(10):
        obs, info = eval_env_final.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_final.step(action)
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

    # =========================================================================
    # STEP 9: Save experiment results to CSV (appends each experiment as a row)
    # =========================================================================
    import csv

    # --- Auto-generate Noted Behavior ---
    BASELINE_AVG_REWARD = 363.3  # <-- Update this with your actual baseline result
    BASELINE_LR = 0.0001
    BASELINE_GAMMA = 0.99
    BASELINE_BATCH = 32
    BASELINE_EPS_END = 0.05
    BASELINE_EPS_DECAY = 0.1

    def generate_noted_behavior(args, avg_reward, best_reward, worst_reward):
        notes = []

        # Compare overall performance to baseline
        diff = avg_reward - BASELINE_AVG_REWARD
        pct = (diff / BASELINE_AVG_REWARD) * 100 if BASELINE_AVG_REWARD != 0 else 0

        if diff > 50:
            notes.append(f"Strong improvement (+{pct:.0f}% over baseline)")
        elif diff > 0:
            notes.append(f"Slight improvement (+{pct:.0f}% over baseline)")
        elif diff > -50:
            notes.append(f"Slight decline ({pct:.0f}% from baseline)")
        elif diff > -200:
            notes.append(f"Significant decline ({pct:.0f}% from baseline)")
        else:
            notes.append(f"Near-random performance ({pct:.0f}% from baseline)")

        # Check stability
        spread = best_reward - worst_reward
        if spread > 300:
            notes.append("Very unstable (high variance between episodes)")
        elif spread > 150:
            notes.append("Moderately stable")
        else:
            notes.append("Very stable and consistent")

        # Explain what the changed hyperparameter did
        if args.lr != BASELINE_LR:
            if args.lr > BASELINE_LR:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Higher lr ({args.lr}) led to faster convergence")
                else:
                    notes.append(f"Higher lr ({args.lr}) caused unstable training, overshot optimal values")
            else:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Lower lr ({args.lr}) gave slower but more stable learning")
                else:
                    notes.append(f"Lower lr ({args.lr}) was too slow to learn in 500k steps")

        if args.gamma != BASELINE_GAMMA:
            if args.gamma < BASELINE_GAMMA:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Lower gamma ({args.gamma}) helped agent focus on immediate rewards effectively")
                else:
                    notes.append(f"Lower gamma ({args.gamma}) made agent too short-sighted to plan ahead")
            else:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Higher gamma ({args.gamma}) helped agent plan further into the future")
                else:
                    notes.append(f"Higher gamma ({args.gamma}) overvalued future rewards, destabilizing updates")

        if args.batch_size != BASELINE_BATCH:
            if args.batch_size > BASELINE_BATCH:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Larger batch ({args.batch_size}) provided more stable gradient estimates")
                else:
                    notes.append(f"Larger batch ({args.batch_size}) led to less frequent updates, slower adaptation")
            else:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Smaller batch ({args.batch_size}) allowed faster, more frequent updates")
                else:
                    notes.append(f"Smaller batch ({args.batch_size}) introduced too much noise in gradient updates")

        if args.epsilon_end != BASELINE_EPS_END or args.epsilon_decay != BASELINE_EPS_DECAY:
            if args.epsilon_decay > BASELINE_EPS_DECAY:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Longer exploration (decay={args.epsilon_decay}) helped discover better strategies")
                else:
                    notes.append(f"Longer exploration (decay={args.epsilon_decay}) spent too long exploring, not enough exploiting")
            if args.epsilon_end > BASELINE_EPS_END:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Higher final epsilon ({args.epsilon_end}) maintained useful exploration")
                else:
                    notes.append(f"Higher final epsilon ({args.epsilon_end}) kept taking too many random actions")
            if args.epsilon_end < BASELINE_EPS_END:
                if avg_reward > BASELINE_AVG_REWARD:
                    notes.append(f"Lower final epsilon ({args.epsilon_end}) committed to learned strategy effectively")
                else:
                    notes.append(f"Lower final epsilon ({args.epsilon_end}) stopped exploring too early, got stuck")

        return ". ".join(notes) + "."

    noted_behavior = generate_noted_behavior(args, avg_reward, best_reward, worst_reward)
    print(f"\n  Noted Behavior: {noted_behavior}")

    csv_file = os.path.join("..", "experiment_results.csv")
    headers = [
        "Experiment", "Member", "lr", "gamma", "batch_size",
        "epsilon_start", "epsilon_end", "epsilon_decay",
        "Avg Reward", "Best Reward", "Worst Reward",
        "Avg Length", "Training Time", "Noted Behavior"
    ]

    # Check if file exists to decide whether to write headers
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            args.experiment,
            args.member_name,
            args.lr,
            args.gamma,
            args.batch_size,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay,
            round(avg_reward, 1),
            round(best_reward, 1),
            round(worst_reward, 1),
            round(avg_length, 1),
            str(training_time),
            noted_behavior,
        ])

    print(f"Results appended to {csv_file}")

    # =========================================================================
    # STEP 10: Print summary for the hyperparameter table
    # =========================================================================
    print("\n" + "=" * 60)
    print("  COPY THIS ROW INTO YOUR HYPERPARAMETER TABLE")
    print("=" * 60)
    print(f"  Experiment:      {args.experiment}")
    print(f"  Member:          {args.member_name}")
    print(f"  Hyperparameters: lr={args.lr}, gamma={args.gamma}, batch_size={args.batch_size}, epsilon_start={args.epsilon_start}, epsilon_end={args.epsilon_end}, epsilon_decay={args.epsilon_decay}")
    print(f"  Avg Reward:      {avg_reward:.1f}")
    print(f"  Best Reward:     {best_reward:.1f}")
    print(f"  Training Time:   {training_time}")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()
    eval_env_final.close()


def run_all_experiments():
    """Run all 10 hyperparameter experiments automatically."""

    MEMBER_NAME = "Fadhl"  # <-- Change this to your name

    # Baseline defaults: lr=0.0001, gamma=0.99, batch_size=32,
    #                    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1

    experiments = [
        # Experiments 1-3: Vary Learning Rate
        {"experiment": 1, "lr": 0.001},
        {"experiment": 2, "lr": 0.00001},
        {"experiment": 3, "lr": 0.0005},

        # Experiments 4-6: Vary Gamma (Discount Factor)
        {"experiment": 4, "gamma": 0.9},
        {"experiment": 5, "gamma": 0.999},
        {"experiment": 6, "gamma": 0.8},

        # Experiments 7-8: Vary Batch Size
        {"experiment": 7, "batch_size": 64},
        {"experiment": 8, "batch_size": 128},

        # Experiments 9-10: Vary Epsilon (Exploration)
        {"experiment": 9, "epsilon_end": 0.1, "epsilon_decay": 0.3},
        {"experiment": 10, "epsilon_end": 0.01, "epsilon_decay": 0.5},
    ]

    # Baseline defaults
    defaults = {
        "lr": 0.0001, "gamma": 0.99, "batch_size": 32,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.1,
        "timesteps": 100000,
    }

    print("=" * 60)
    print(f"  RUNNING ALL {len(experiments)} EXPERIMENTS FOR {MEMBER_NAME}")
    print(f"  Estimated time: ~{len(experiments) * 50} minutes")
    print("=" * 60)

    results = {}
    for exp in experiments:
        exp_num = exp["experiment"]

        # Build args: start with defaults, then override with this experiment's changes
        exp_args = argparse.Namespace(
            member_name=MEMBER_NAME,
            **{**defaults, **exp}
        )

        print("\n" + "=" * 60)
        print(f"  STARTING EXPERIMENT {exp_num} of {len(experiments)}")
        print(f"  Changes from baseline: {exp}")
        print("=" * 60 + "\n")

        try:
            train(exp_args)
            results[exp_num] = "PASSED"
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_num} FAILED: {e}")
            results[exp_num] = "FAILED"

    # Print final summary
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS FINISHED")
    print("=" * 60)
    for exp_num, status in results.items():
        print(f"  Experiment {exp_num:2d}: {status}")
    print("=" * 60)
    print(f"\n  Check the 'logs' folder for detailed results.")
    print(f"  Check the 'models' folder for saved models.")


if __name__ == "__main__":
    args = parse_args()

    if args.run_all:
        run_all_experiments()
    else:
        train(args)
