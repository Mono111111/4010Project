from environment.gym_env import SnakeGymEnv
from agents.sarsa import SARSA_Agent

# Train SARSA agent
if __name__ == "__main__":
	#env = SnakeGymEnv(render_mode="human", max_steps=2000)
	env = SnakeGymEnv(render_mode=None, max_steps=2000)
	agent = SARSA_Agent(env, max_episode = 500)
	history = agent.SARSA()
	print("Training completed.")