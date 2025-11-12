import numpy as np
import math
import pickle
from collections import defaultdict

# Return the action with the highest Q value for a given state
# if has many max_q, choose one randomly
def argmax_Q(Q, state):
	q_values = [Q[(state, a)] for a in range(4)]
	max_q = max(q_values)
	optimal_actions = [a for a in range(4) if q_values[a] == max_q]
	return np.random.choice(optimal_actions)

def sign3(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

# 将连续state离散化
def discretize_state(obs):
	hx, hy, energy, score, dx, dy = map(float, obs)
	x_cell = int(hx // 20)
	y_cell = int(hy // 20)
	sdx = sign3(dx)
	sdy = sign3(dy)
	energy_bin = int(energy // 10)
	return (x_cell, y_cell, sdx, sdy, energy_bin)

# Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
class SARSA_Agent:
	# Initialize
	# 果然还是需要一个init看的清楚一点
	def __init__(self, env, alpha = 0.15, gamma=0.9, step_size=0.1, epsilon=1.0, max_episode=1000, epsilon_decay=0.995, epsilon_min=0.01):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.step_size = step_size
		self.epsilon = epsilon
		self.max_episode = max_episode
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.Q = defaultdict(float)
		# 4
		self.n_actions = env.action_space.n
	# actions: number of possible actions
	# alpha: learning rate
	# gamma: discount factor
	# epsilon: exploration rate at begining
	# max_episode: number of episodes to train
	# epsilon_decay: decay rate of exploration rate per episode 探索衰减率
	# epsilon_min: minimum exploration rate
	def SARSA(env, gamma, step_size, epsilon, max_episode, epsilon_decay = 0.995, epsilon_min=0.01):
		#training history
		episode_scores = []
		episode_steps = []
		episode_rewards = []

		for episode in range(max_episode):
			#init
			state = env.reset()
			terminated = False
			truncated = False
			step = 0
			max_steps = 2000
			episode_reward = 0

			#choose action using epsilon-greedy policy
			if np.random.random()< epsilon:
				action = np.random.randint(4)
			else:
				action = argmax_Q(Q, state)
			
			while not (terminated or truncated) and step < max_steps:
				next_state, reward, terminated, truncated, info = env.step(action)
				next_state = tuple(next_state)

				if terminated or truncated:
					Q[(state, action)] += step_size * (reward - Q[(state, action)])
					break
				else:
					if np.random.random() < epsilon:
						next_action = np.random.randint(4)
					else:
						next_action = argmax_Q(Q, next_state)
					
					target = reward + gamma * Q[(next_state, next_action)]
					Q[(state, action)] += step_size * (target - Q[(state, action)])
					state = next_state
					action = next_action
					step += 1
		policy = np.zeros((env.observation_space.n, ), dtype=int)
		for s in range(env.observation_space.n):
			policy[s] = argmax_Q(Q, s)
		Pi = diagonalization(policy, state, action)
		q_values = Q.reshape(-1)
		return Pi,q_values