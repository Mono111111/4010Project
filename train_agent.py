from environment.gym_env import SnakeGymEnv
from agents.sarsa import SARSA_Agent
from agents.agent import Greedy_Agent
from agents.q_learning import QLearningAgent
import numpy as np

# Train SARSA agent
# can also be use in other agent
#change the line 15-19 mainly
# Test SARSA agent with agent.py use the saved q table
if __name__ == "__main__":
	
	# Training
	#env = SnakeGymEnv(render_mode="human", max_steps=2000)
	env = SnakeGymEnv(render_mode=None, max_steps=2000)

	# �����޸Ŀ��Ի����agent����
	agent = QLearningAgent(env, max_episode = 500)
	history = agent.SARSA()
	# Save the trained Q-table
	agent.save("sarsa_q_table.pkl")
	print("SARSA Training completed.")

	rewards = history['episode_rewards']
	steps = history['episode_steps']
	# Use last 50 episodes
	last_n = 50
	avg_reward = np.mean(rewards[-last_n:])
	avg_steps = np.mean(steps[-last_n:])

	# Use greedy agent for evaluation
	eval_episodes = 50
	greedy_agent = Greedy_Agent(env, model_path="sarsa_q_table.pkl")
	eval_rewards = []
	eval_steps = []
	for ep in range(eval_episodes):
		obs, info = env.reset()
		terminated = False
		truncated = False
		total_reward = 0.0
		step = 0
		while not (terminated or truncated):
			action = greedy_agent.select_action(obs)
			obs, reward, terminated, truncated, info = env.step(action)
			total_reward += reward
			step += 1
		eval_rewards.append(total_reward)
		eval_steps.append(step)

	avg_greedy_reward = np.mean(eval_rewards)
	avg_greedy_steps = np.mean(eval_steps)

	print("\n---Overall---")
	print(f"[Training] Last {last_n} episodes:  avg reward = {avg_reward:.2f}, avg steps = {avg_steps:.1f}")
	print(f"[Evaluation] {eval_episodes} episodes: avg reward = {avg_greedy_reward:.2f}, avg steps = {avg_greedy_steps:.1f}")