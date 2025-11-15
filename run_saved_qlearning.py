"""
Run a trained Q-Learning agent in the Snake environment.
Make sure 'qlearning_q_table.pkl' exists before running.
"""

import time
import numpy as np
from environment.gym_env import SnakeGymEnv
from agents.q_learning import QLearningAgent

# ===================================================
# 1️⃣ 初始化环境与智能体
# ===================================================
env = SnakeGymEnv(render_mode="human", max_steps=2000)
agent = QLearningAgent(
    action_space=env.action_space,
    state_shape=env.observation_space.shape
)

# ===================================================
# 2️⃣ 加载已训练好的模型
# ===================================================
try:
    agent.load("qlearning_q_table.pkl")
    print("[INFO] Loaded Q-table from 'qlearning_q_table.pkl'")
except FileNotFoundError:
    print("[ERROR] qlearning_q_table.pkl not found!")
    print("Please train the model first using train_agent.py with mode='qlearning'.")
    exit()

# ===================================================
# 3️⃣ 运行（演示智能体）
# ===================================================
num_episodes = 5   # 运行几轮演示
for ep in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        time.sleep(0.03)  # 控制可视化速度（0.03秒一帧）

    print(f"[EP {ep+1}] Finished | Steps={step}, Total Reward={total_reward:.1f}")

env.close()
print("\n✅ Demo finished successfully.")
