# 4010Project
4010 group project

proposal: https://docs.google.com/document/d/1LLpeT19tY8b8aanu7BanDeNcpHKp44U__Y7EoGTkEIc/edit?tab=t.0#heading=h.rd2ebpiuo8v4

Overview:
This project implements and compares three reinforcement learning algorithms: 
Q-Learning, SARSA, and Deep Q-Network (DQN), using a custom Snake environment

The goal is to train an intelligent snake agent that learns to survive longer and maximize rewards through exploration and experience-based updates.


This repository contains:
- Keyboard version: python main.py 
- Gym Random Agent: python random_agent_demo.py
- Full implementations of all three algorithms
- A custom Gym-like environment (`SnakeGymEnv`)
- Visualization and performance comparison scripts

Project Structure:
- agents/: Contains implementations of Q-Learning, SARSA, and DQN agents.
q_learning.py: Q-Learning agent implementation.
sarsa.py: SARSA agent implementation.
dqn.py: Deep Q-Network agent implementation.
init.py: Initializes the agents package.

- environment/: Contains the custom Snake environment.
config.py: Configuration settings for the environment.
core.py: Core environment logic.
env.py: Environment interface.
gym_env.py: Gym-compatible wrapper.
renderer.py: Rendering (Pygame)
init.py: Initializes the environment package.

- compare.py: Unified comparison of Q-Learning, SARSA, DQN.
- main.py: ntry point for running agents.
- play_agent.py: Replay a trained agent.
- random_agent_demo.py: Baseline random agent demo

- train_qlearning.py: Train or test Q-Learning agent
- train_sarsa.py: Train or test SARSA agent
- train_dqn.py: Train or test DQN agent

- requirements.txt: Python dependencies.
- README.md: Project documentation.

Getting Started:
1. Clone the repository:
   git clone
2. Install dependencies:
   pip install -r requirements.txt
3. Run the desired training or demo script, e.g.:
   python train_qlearning.py
   python random_agent_demo.py
   python compare.py
