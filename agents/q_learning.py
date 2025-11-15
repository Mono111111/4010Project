# agents/q_learning.py
import numpy as np
import pickle
from collections import defaultdict
from environment import config

class QLearningAgent:
    def __init__(
        self,
        env,
        alpha=0.15,           # learning rate
        gamma=0.9,          # discount factor
        epsilon=1.0,         # exploration rate
        epsilon_min=0.01,    # minimum exploration rate
        epsilon_decay=0.995, # exploration decay rate
        episodes=1000,       # number of training episodes
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        
        # Q-table: defaultdict with default value of zeros for each action
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Action space size
        self.n_actions = env.action_space.n
    
    def get_state(self):
        """
        Convert continuous observation to discrete state.
        We discretize the observation space to make Q-learning tractable.
        """
        obs = self.env.unwrapped.game.get_observation()
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
        direction = self.env.unwrapped.game.direction
        if direction == [0, -20]:  # UP
            dir_idx = 0
        elif direction == [0, 20]:  # DOWN
            dir_idx = 1
        elif direction == [-20, 0]:  # LEFT
            dir_idx = 2
        else:  # RIGHT
            dir_idx = 3
        
        # Check if there's danger in front (obstacle, body, dynamic obstacle)
        game = self.env.unwrapped.game
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
    
    def get_q_values(self, state):
        """Get Q-values for all actions in given state."""
        return self.q_table[state]
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: best action based on Q-values
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule:
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        """
        current_q = self.q_table[state][action]
        
        if done:
            # No future rewards if episode is done
            target_q = reward
        else:
            # Future reward is max Q-value of next state
            max_next_q = np.max(self.get_q_values(next_state))
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def train(self):
        """
        Train the Q-learning agent.
        Returns list of episode rewards.
        """
        episode_rewards = []
        
        for episode in range(self.episodes):
            obs, info = self.env.reset(seed=episode)
            state = self.get_state()
            
            episode_reward = 0
            done = False
            steps = 0
            prev_energy = info['energy']  # Track energy for collision detection
            
            while not done:
                # Choose action
                action = self.choose_action(state)
                
                # Record old head position for reward shaping
                old_head_x, old_head_y, _, _, _, _ = self.env.unwrapped.game.get_observation()
                
                # Take action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Additional penalty for collision (energy drop detection)
                curr_energy = info['energy']
                if curr_energy < prev_energy - 5:  # Energy dropped significantly
                    reward -= 10  # Extra penalty for hitting obstacle/self-bite
                prev_energy = curr_energy
                
                # Reward shaping: encourage moving toward food
                #new_head_x, new_head_y, _, _, _, _ = self.env.unwrapped.game.get_observation()
                #foods = self.env.unwrapped.game.core.foods
                #fx, fy, _ = min(foods, key=lambda f: abs(f[0]-old_head_x)+abs(f[1]-old_head_y))
                
                #old_dist = abs(fx - old_head_x) + abs(fy - old_head_y)
                #new_dist = abs(fx - new_head_x) + abs(fy - new_head_y)
                
                #if new_dist < old_dist:
                #    reward += 0.2  # bonus for getting closer
                #else:
                #    reward -= 0.2  # penalty for getting farther
                
                # Get next state
                next_state = self.get_state()
                
                # Update Q-table
                self.update_q_value(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            episode_rewards.append(episode_reward)
            
            # Print progress every 50 episodes
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode + 1}/{self.episodes} | "
                      f"Avg Reward (last 50): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {steps} | "
                      f"Score: {info['score']}")
        
        print(f"\n✅ Training completed! Total episodes: {self.episodes}")
        print(f"📊 Q-table size: {len(self.q_table)} states")
        
        return episode_rewards
    
    def save(self, filename):
        """Save Q-table to file."""
        data = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Q-table saved to {filename}")
    
    def load(self, filename):
        """Load Q-table from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        
        print(f"📂 Q-table loaded from {filename}")
        print(f"📊 Q-table size: {len(self.q_table)} states")