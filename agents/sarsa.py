import numpy as np
import math
import pickle
from collections import defaultdict

# Return the action with the highest Q value for a given state
# if has many max_q, choose one randomly
def argmax_Q(Q, state, n_actions):
	q_values = [Q[(state, a)] for a in range(n_actions)]
	max_q = max(q_values)
	optimal_actions = [a for a in range(n_actions) if q_values[a] == max_q]
	return np.random.choice(optimal_actions)

def sign3(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

# 将连续state离散化成元组
def discretize_state(obs):
	hx, hy, energy, score, dx, dy = map(float, obs)
	# position cell size 20
	# calcaluate the x and y cell of head
	x_cell = int(hx // 20)
	y_cell = int(hy // 20)
	# left -1, no change 0, right 1
	sdx = sign3(dx)
	sdy = sign3(dy)
	# energy bin size 10
	energy_bin = int(energy // 10)
	# return as a tuple
	return (x_cell, y_cell, sdx, sdy, energy_bin)

# Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
class SARSA_Agent:
	# Initialize
	# 果然还是需要一个init看的清楚一点
	def __init__(self, env, alpha = 0.15, gamma=0.9, epsilon=1.0, max_episode=1000, epsilon_decay=0.995, epsilon_min=0.01, max_steps = 2000):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.max_episode = max_episode
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.max_steps = max_steps
		# get the action space in env action space
		self.n_actions = env.action_space.n
		# Q table
		self.Q = defaultdict(float)
	# actions: number of possible actions
	# alpha: learning rate
	# gamma: discount factor
	# epsilon: exploration rate at begining
	# max_episode: number of episodes to train
	# epsilon_decay: decay rate of exploration rate per episode 探索衰减率
	# epsilon_min: minimum exploration rate

	# ε-greedy
	def _epsilon_greedy(self, state):
		# choose action randomly
		if np.random.random() < self.epsilon:
			return np.random.randint(self.n_actions)
		# choose action with highest Q value
		else:
			return argmax_Q(self.Q, state, self.n_actions)

	def SARSA(self):
		#training history
		#episode_scores = []
		episode_steps = []
		episode_rewards = []

		for ep in range(self.max_episode):
			#init
			# recheck the env
			# reset() return two values obs and info
			obs, info = self.env.reset()
			# some copy from assignemnt
			terminated = False
			truncated = False
			state = discretize_state(obs)
			action = self._epsilon_greedy(state)
			step = 0
			total_reward = 0.0
			
			while not (terminated or truncated) and step < self.max_steps:
				step += 1
				next_obs, reward, terminated, truncated, info = self.env.step(action)
				next_state = discretize_state(next_obs)
				total_reward += reward

				if terminated or truncated:
					td_target = reward
					td_error = td_target - self.Q[(state, action)]
					self.Q[(state, action)] += self.alpha * td_error
					break
				else:
					next_action = self._epsilon_greedy(next_state)
					# R + γ* Q(s', a')
					td_target = reward + self.gamma * self.Q[(next_state, next_action)]
					td_error = td_target - self.Q[(state, action)]
					# update Q
					self.Q[(state, action)] += self.alpha * td_error
					# move to next state and action
					state = next_state
					action = next_action
			self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
			#episode_scores.append(info['score'])
			# add history
			episode_steps.append(step)
			episode_rewards.append(total_reward)

			# print to obeserve
			# 10 episodes print once
			if(ep+1) % 10 == 0:
				print(f"Episode {ep+1}/{self.max_episode}, Steps: {step}, Total Reward: {total_reward:.1f}")

		history = {"episode_rewards": episode_rewards, "episode_steps": episode_steps, "Q" : self.Q}
		return history