import numpy as np
import math
import pickle
from environment import config
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

def discretize_state(obs):
	hx, hy, energy, score, dx, dy = map(float, obs)
	# energy bin size
	energy_bin = min(int(energy / 20), 5)
	if abs(dx) <20 and abs(dy) <20:
		food_dir = 0
	else:
		angle = np.arctan2(dy, dx)
		food_dir = int((angle + np.pi) / (2 * np.pi / 8)) +1
	direction = env.unwrapped.game.direction
	if direction == [0, -20]:   # up
		dir_idx = 0
	elif direction == [0, 20]:  # down
		dir_idx = 1
	elif direction == [-20, 0]: # left
		dir_idx = 2
	else:
	    dir_idx = 3               # right

	game = env.unwrapped.game
	next_x = (hx + direction[0]) % 600
	next_y = (hy + direction[1]) % 600

	danger_ahead = 0
	if [next_x, next_y] in game.core.snake[1:]:
		danger_ahead = 1
	elif any(next_x == ox and next_y == oy for ox, oy in game.core.obstacles):
		danger_ahead = 1
	elif any(next_x == dx_ and next_y == dy_ for dx_, dy_, _ in game.core.dynamic_obstacles):
		danger_ahead = 1

	return (energy_bin, food_dir, dir_idx, danger_ahead)

def get_state(env):
        obs = env.unwrapped.game.get_observation()
        head_x, head_y, energy, score, dx, dy = obs
        
        # Discretize energy into bins (more granular than before)
        energy_bin = min(int(energy / 20), 5)  # 0-5 bins (better granularity)
        
        # Discretize food direction into 8 directions
        if abs(dx) < 5 and abs(dy) < 5:
            food_dir = 0  # very close
        else:
            angle = np.arctan2(dy, dx)
            food_dir = int((angle + np.pi) / (2 * np.pi / 8)) + 1  # 1-8
        
        # Get current direction
        direction = env.unwrapped.game.direction
        if direction == [0, -20]:  # UP
            dir_idx = 0
        elif direction == [0, 20]:  # DOWN
            dir_idx = 1
        elif direction == [-20, 0]:  # LEFT
            dir_idx = 2
        else:  # RIGHT
            dir_idx = 3
        
        # Check if there's danger in front (obstacle, body, dynamic obstacle)
        game = env.unwrapped.game
        next_x = (head_x + direction[0]) % config.WINDOW_WIDTH
        next_y = (head_y + direction[1]) % config.WINDOW_HEIGHT
        
        danger_ahead = 0
        # Check body collision
        if [next_x, next_y] in game.core.snake[1:]:
            danger_ahead = 1
        # Check static obstacles
        elif any(next_x == ox and next_y == oy for ox, oy in game.core.obstacles):
            danger_ahead = 1
        # Check dynamic obstacles
        elif any(next_x == dx and next_y == dy for dx, dy, _ in game.core.dynamic_obstacles):
            danger_ahead = 1
        
        # Create state tuple (now with danger detection)
        state = (energy_bin, food_dir, dir_idx, danger_ahead)
        return state

# Q(s,a)
class SARSA_Agent:
	# Initialize
	def __init__(self, env, alpha = 0.15, gamma=0.9, epsilon=1.0, episodes=1000, epsilon_decay=0.995, epsilon_min=0.01):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.episodes = episodes
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		#self.max_steps = max_steps
		# get the action space in env action space
		self.n_actions = env.action_space.n
		# Q table
		self.Q = defaultdict(float)
	# actions: number of possible actions
	# alpha: learning rate
	# gamma: discount factor
	# epsilon: exploration rate at begining
	# max_episode: number of episodes to train
	# epsilon_decay: decay rate of exploration rate per episode ̽��˥����
	# epsilon_min: minimum exploration rate

	# ��-greedy
	def _epsilon_greedy(self, state):
		# choose action randomly
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		# choose action with highest Q value
		else:
			return argmax_Q(self.Q, state, self.n_actions)

	def update_q_value(self, state, action, reward, next_state, next_action, done):
		current_q = self.Q[(state, action)]
		if done:
			target_q = reward
		else:
			next_q = self.Q[(next_state, next_action)]
			target_q = reward + self.gamma * next_q
		self.Q[(state, action)] = current_q + self.alpha * (target_q - current_q)

	def SARSA(self):
		#training history
		#episode_scores = []
		episode_steps = []
		episode_rewards = []

		for ep in range(self.episodes):
			#init
			# recheck the env
			# reset() return two values obs and info
			obs, info = self.env.reset(seed = ep)
			# some copy from assignemnt
			terminated = False
			truncated = False
			state = get_state(self.env)
			action = self._epsilon_greedy(state)
			done = False
			prev_energy = info['energy']
			step = 0
			total_reward = 0.0
			
			while not done:
				obs, reward, terminated, truncated, info = self.env.step(action)
				done = terminated or truncated
				curr_energy = info['energy']
				if curr_energy < prev_energy -5:
					reward -=10
				prev_energy = curr_energy
				next_state = get_state(self.env)
				next_action = self._epsilon_greedy(next_state)
				self.update_q_value(state, action, reward, next_state, next_action, done)
				state = next_state
				action = next_action
				total_reward += reward
				step += 1

			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay
			episode_rewards.append(total_reward)

			# print to obeserve
			# 10 episodes print once
			if(ep+1) % 50 == 0:
				avg_reward = np.mean(episode_rewards[-50:])
				print(f"Episode {ep + 1}/{self.episodes} | "
				      f"Avg Reward (last 50): {avg_reward:.2f} | "
				      f"Epsilon: {self.epsilon:.3f} | "
				      f"Steps: {step} | "
				      f"Score: {info['score']}")

		print(f"\n Training completed! Total episodes: {self.episodes}")
		print(f"Q-table size: {len(self.Q)} Q-values")
		return episode_rewards

	def save(self, path):
		data = {
			'Q': dict(self.Q),
			'alpha': self.alpha,
			'gamma': self.gamma,
			'epsilon': self.epsilon,
			'epsilon_min': self.epsilon_min,
			'epsilon_decay': self.epsilon_decay,
		}
		with open(path, 'wb') as f:
			pickle.dump(data, f)

	def load(self, path):
		with open(path, 'rb') as f:
			data = pickle.load(f)
		self.Q = defaultdict(float, data['Q'])
		self.alpha = data['alpha']
		self.gamma = data['gamma']
		self.epsilon = data['epsilon']
		self.epsilon_min = data['epsilon_min']
		self.epsilon_decay = data['epsilon_decay']