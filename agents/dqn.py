# agents/dqn.py
import random
import numpy as np
from collections import deque
from environment import config

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    # Simple 2-layer fully connected network mapping state → Q-values
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)



class ReplayBuffer:
    # Replay buffer storing (s, a, r, s', done) transitions
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_next, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQN_Agent:
    def get_state(self):
        obs = self.env.unwrapped.game.get_observation()
        head_x, head_y, energy, score, dx, dy = obs
        energy_bin = min(int(energy / 20), 5)
        energy_features = np.zeros(6, dtype=np.float32)
        energy_features[energy_bin] = 1.0

        if abs(dx) <5 and abs(dy) <5:
            food_dir = 0
        else:
            angle = np.arctan2(dy, dx)
            food_dir = min(int((angle + np.pi) / (2 * np.pi / 8)) + 1, 8)
        food_dir_features = np.zeros(9, dtype=np.float32)
        food_dir_features[food_dir] = 1.0

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
        dir_features = np.zeros(4, dtype=np.float32)
        dir_features[dir_idx] = 1.0
        
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
        return np.concatenate([energy_features, food_dir_features, dir_features, [danger_ahead]])

    # DQN agent using Q-network, target network, replay buffer, and epsilon-greedy
    def __init__(self, env,
                 gamma=0.99,
                 batch_size=64,
                 lr=1e-3,
                 replay_capacity=100_000,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=50_000,
                 max_steps_per_episode=2000,
                 hidden_dim=128,
                 target_update_freq=500,
                 device="cpu"):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_steps = max_steps_per_episode

        # State & action dimensions
        self.state_dim = 20
        self.n_actions = env.action_space.n

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Q-network & target network
        self.q_net = QNetwork(self.state_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_capacity = replay_capacity
        self.replay = ReplayBuffer(capacity=self.replay_capacity)

        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        # Target network update frequency
        self.learn_steps = 0
        self.target_update_freq = target_update_freq

    def _current_epsilon(self):
        # Linearly decay epsilon over time
        frac = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _epsilon_greedy(self, state, greedy=False):
       # Return action using epsilon-greedy (greedy=True disables exploration)
        if (not greedy) and random.random() < self._current_epsilon():
            return random.randint(0, self.n_actions - 1)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def _update_network(self):
        # Sample a batch from replay buffer and update Q-network.
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_net(states_t).gather(1, actions_t)

        # Target y = r + γ * max Q_target(s', a')
        with torch.no_grad():
            next_q_online = self.q_net(next_states_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B,1)

            next_q_target = self.target_net(next_states_t)
            next_q_values = next_q_target.gather(1, next_actions)      # (B,1)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q_values


        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self, max_episode=500):
        # Train DQN and return training history
        episode_rewards = []
        episode_steps = []

        for ep in range(max_episode):
            obs, info = self.env.reset()
            state = self.get_state()
            done = False
            truncated = False
            step = 0
            total_reward = 0.0

            prev_energy = info.get("energy", None)

            while (not done) and (not truncated) and step < self.max_steps:
                step += 1
                self.total_steps += 1

                action = self._epsilon_greedy(state, greedy=False)
                next_obs, reward, done, truncated, info = self.env.step(action)
                next_state = self.get_state()
                curr_energy = info.get("energy", None)
                if prev_energy is not None and curr_energy is not None:
                    if curr_energy < prev_energy - 5:
                        reward -= 10.0
                prev_energy = curr_energy
                
                total_reward += reward

                done_flag = float(done or truncated)

                self.replay.push(state, action, reward, next_state, done_flag)
                self._update_network()

                state = next_state

            episode_rewards.append(total_reward)
            episode_steps.append(step)

            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(
                    f"Episode {ep+1}/{max_episode}, "
                    f"Steps: {step}, Reward: {total_reward:.1f}, "
                    f"Avg10: {avg_reward:.1f}, "
                    f"Epsilon: {self._current_epsilon():.3f}"
                )

        return {"episode_rewards": episode_rewards, "episode_steps": episode_steps}

    def save(self, path="dqn_snake.pt"):
        # Save Q-network parameters
        torch.save(self.q_net.state_dict(), path)
        print(f"DQN model saved to {path}")

    def load(self, path="dqn_snake.pt"):
        # Load Q-network parameters
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        print(f"DQN model loaded from {path}")

    def evaluate(self, n_episodes=20, render=False):
        # Evaluate agent using greedy policy
        rewards = []
        steps_list = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            state = self.get_state()
            done = False
            truncated = False
            step = 0
            total_reward = 0.0

            while (not done) and (not truncated) and step < self.max_steps:
                step += 1
                action = self._epsilon_greedy(state, greedy=True)
                obs, reward, done, truncated, info = self.env.step(action)
                state = self.get_state()
                total_reward += reward

                if render:
                    pass

            rewards.append(total_reward)
            steps_list.append(step)

        avg_reward = float(np.mean(rewards))
        avg_steps = float(np.mean(steps_list))
        print(f"[Eval] episodes={n_episodes}, avg reward={avg_reward:.2f}, avg steps={avg_steps:.1f}")
        return avg_reward, avg_steps


# Standalone training entry
if __name__ == "__main__":
    from environment.gym_env import SnakeGymEnv

    env = SnakeGymEnv(render_mode=None, max_steps=2000)
    agent = DQN_Agent(env, max_steps_per_episode=2000)

    history = agent.train(max_episode=500)
    agent.save("dqn_snake.pt")

    agent.evaluate(n_episodes=50, render=False)

    print("\nRunning one greedy episode with rendering...")
    eval_env = SnakeGymEnv(render_mode="human", max_steps=2000)
    agent.env = eval_env
    agent.evaluate(n_episodes=1, render=True)
    eval_env.close()
