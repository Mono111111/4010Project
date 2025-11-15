import numpy as np
import pickle
from collections import defaultdict

# copy from sarsa

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

class Greedy_Agent:
	def __init__(self, env, model_path=None):
		self.env = env
		self.n_actions = env.action_space.n
		self.Q = defaultdict(float)
		if model_path is not None:
			self.load(model_path)

	def select_action(self, obs):
		state = discretize_state(obs)
		action = argmax_Q(self.Q, state, self.n_actions)
		return action

	def load(self, model_path):
		with open(model_path, 'rb') as f:
			data = pickle.load(f)
		self.Q = defaultdict(float, data)