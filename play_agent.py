from environment.gym_env import SnakeGymEnv
from agents.sarsa import SARSA_Agent
from agents.agent import Greedy_Agent

if __name__ == "__main__":
	#env = SnakeGymEnv(render_mode="human", max_steps=2000)
	env = SnakeGymEnv(render_mode=None, max_steps=2000)
	agent = Greedy_Agent(env, model_path = "sarsa_q_table.pkl")
	obs, info = env.reset()
	terminated = False
	truncated = False
	while not(terminated or truncated):
		action = agent.select_action(obs)
		obs, reward, terminated, truncated, info = env.step(action)