# Deep Q-Network (DQN) Agent for Atari Assault

## Project Overview

This project implements a **Deep Q-Network (DQN)** agent to play the Atari game **Assault** using **Python**, **Stable Baselines3 (SB3)**, and **Gymnasium**. The main objective was to explore how different hyperparameter settings (learning rate, discount factor, batch size, and exploration strategy) affect the agent's performance and stability.

Key goals:
- Train a DQN agent to maximize average reward in Assault.
- Evaluate multiple hyperparameter combinations.
- Select the best-performing model for gameplay demonstrations.

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

- Selected Model: 
- Hyperparameters:
- Average Reward: 
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
