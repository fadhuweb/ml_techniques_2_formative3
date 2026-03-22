# Deep Q-Network (DQN) Agent for Atari Assault

## Project Overview

This project implements a **Deep Q-Network (DQN)** agent to play the Atari game **Assault** using **Python**, **Stable Baselines3 (SB3)**, and **Gymnasium**. The main objective was to explore how different hyperparameter settings (learning rate, discount factor, batch size, and exploration strategy) affect the agent's performance and stability.

Key goals:
- Train a DQN agent to maximize average reward in Assault.
- Evaluate multiple hyperparameter combinations.
- Select the best-performing model for gameplay demonstrations.

---
## Choice of Policy: CNN vs MLP

In our experiments, we evaluated **two types of policies** in the **DQN framework** for the `ALE/Assault-v5` environment:  

1. **MlpPolicy** – A fully connected multi-layer perceptron (MLP) that takes raw RAM observations (128 bytes) as input.  
2. **CnnPolicy** – A convolutional neural network (CNN) that processes the raw visual frames (210x160x3) from the Atari environment.

### Evaluation Results

| Metric | MlpPolicy (RAM) | CnnPolicy (Frames) |
|--------|-----------------|------------------|
| Average Reward | 363.3 | 518.7 |
| Best Reward | 462.0 | 609.0 |
| Worst Reward | 231.0 | 378.0 |
| Average Episode Length | 642.6 | 1764.0 |

**Observations:**  
- The **CNN policy significantly outperformed** the MLP policy across all metrics.  
- With CNN, the agent achieved a higher **average reward (+43%)**, a higher **best reward**, and much longer **episode lengths**, indicating better strategy and survival.  
- MLP on RAM struggled to capture the spatial and temporal patterns of the game, which limited performance despite consistent training.

**Conclusion:**  
The CNN policy is clearly superior for frame-based Atari environments. Therefore, all further experiments and the final playing agent use `CnnPolicy` to maximize performance.

---
## Environment & Tools

- **Python**: 3.10+  
- **Stable Baselines3**: 2.6+  
- **Gymnasium**: 0.29+  
- **ALE-Py (Arcade Learning Environment)**: 0.11+  
- **OpenCV**: Required for image preprocessing (`pip install opencv-python`)  
- **Other Libraries**: NumPy, CSV, OS, datetime

**Environment Setup Example:**

```bash
pip install stable-baselines3[extra] gymnasium ale-py opencv-python numpy
```
### Game Environment:
- ALE/Assault-v5 (Atari 2600 game): Observations are processed using AtariWrapper to standardize input size (84,84,1) and apply frame stacking.

## Hyperparameter Experiments

A total of 40 experiments were conducted to explore different combinations of hyperparameters. All experiments used:

- Policy: CnnPolicy (to process visual inputs)
- Buffer Size: 50,000
- Learning Starts: 1,000 steps
- Target Update Interval: 1,000 steps
- Train Frequency: every 4 steps
- All experiments and results are documented [here](https://docs.google.com/spreadsheets/d/1E7JWT3_4jDCHLWUFGVtR_gzomi6mxVNZyZ_mFhYhvto/edit?usp=sharing)

## Best Model

- Selected Model: Jinelle's Experiment 10 (models/best_experiment.zip)
- Hyperparameters: lr=0.0005, gamma=0.95, batch=64, eps_start=1.0, eps_end=0.1, eps_frac=0.2
- Average Reward: 474.60
- Reason for Selection: This model achieved the highest average reward by balancing learning speed, future reward estimation, and exploration-exploitation tradeoff.
- Demo video showing agent playing in the Atari environment [here](https://drive.google.com/drive/folders/1EJ9PdQBcQeIDBgYKYJ2rjRa1SobF9cJa?usp=drive_link)

## How to Play

Use play.py to load and visualize the trained agent:

```python play.py```

Notes:
The script automatically uses DummyVecEnv and VecTransposeImage to match the training preprocessing.
Gameplay will render in a GUI window.
The agent plays using a greedy policy, always selecting the action with the highest predicted Q-value.

## Contributors
- Jinelle Nformi
- Wenebi Fiderikumo
- Fadhlullah Abdulazeez
- Emmanuel Dania
