import numpy as np
import math
from collections import defaultdict

class SARSA_Agent:
	// Initialize
	def _init_(self, actions, alpha=0.1, gamma=0.9, epsilon_start = 0.30, epsilon_end = 0.02. epsilon_steps = 50_000, energy_coef = -0.1):
		self.actions = actions
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_steps = epsilon_steps
		self.energy_coef = energy_coef
		self.stpes_done = 0
		self.Q = defaultdict(lambda: np.zeros(actions, dtype = np.float32))
