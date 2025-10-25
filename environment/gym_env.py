import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environment.env import SnakeGame
from environment import config

# Discrete action mapping used by both human control and agents:
# 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
ACTION2DIR = {
    0: [0, -config.GRID_SIZE],   # UP
    1: [0,  config.GRID_SIZE],   # DOWN
    2: [-config.GRID_SIZE, 0],   # LEFT
    3: [ config.GRID_SIZE, 0],   # RIGHT
}

class SnakeGymEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None, max_steps=2000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.game = SnakeGame()  # contains all internal logic and pygame window

        # Discrete actions (UP/DOWN/LEFT/RIGHT)
        self.action_space = spaces.Discrete(4)

        # Observation space bounds (match get_observation())
        high = np.array([
            config.WINDOW_WIDTH,          # head_x
            config.WINDOW_HEIGHT,         # head_y
            100.0,                        # energy
            1000.0,                       # score (loose upper bound for Box)
            config.WINDOW_WIDTH,          # dx (can be negative -> see low)
            config.WINDOW_HEIGHT          # dy (can be negative -> see low)
        ], dtype=np.float32)
        low = np.array([
            0.0, 0.0,                     # head_x, head_y
            0.0, 0.0,                     # energy, score
            -config.WINDOW_WIDTH,         # dx
            -config.WINDOW_HEIGHT         # dy
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.steps = 0  # step counter for truncation

    def _obs(self):
        # SnakeGame.get_observation() returns a 6-tuple; convert to np.array
        return np.array(self.game.get_observation(), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Make randomness reproducible
        super().reset(seed=seed)
        if seed is not None:
            # keep Python RNG aligned for consistent food/obstacle placement
            import random
            random.seed(seed)
            # If you'd like, also seed numpy RNG here:
            # np.random.seed(seed)

        self.game.reset()
        self.steps = 0
        obs = self._obs()
        info = {
            "score": self.game.score,
            "energy": self.game.energy,
            "mystery_foods": sum(1 for f in getattr(self.game, "foods", []) if f[2] == "mystery"),
        }
        return obs, info

    def step(self, action: int):
        
        # Current direction (to enforce "no direct reversal")
        cur_dx, cur_dy = self.game.direction
        new_dx, new_dy = ACTION2DIR[int(action)]

        # Disallow exact opposite moves (UP<->DOWN, LEFT<->RIGHT)
        if not (cur_dx == -new_dx and cur_dy == -new_dy):
            self.game.direction = [new_dx, new_dy]

        # Advance one tick; SnakeGame will internally compute self.last_reward and game_over
        self.game.step()
        self.steps += 1

        obs = self._obs()
        reward = float(getattr(self.game, "last_reward", 0.0))
        terminated = bool(self.game.game_over)      # natural end (energy<=0, score target, etc.)
        truncated = self.steps >= self.max_steps    # artificial cutoff

        if self.render_mode == "human":
            self.game.render()

        info = {
            "score": self.game.score,
            "energy": self.game.energy,
            "steps": self.steps,
            "mystery_foods": sum(1 for f in getattr(self.game, "foods", []) if f[2] == "mystery"),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # Render only when in human mode (avoid accidental drawing during training)
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        # Nothing special to clean for now
        pass
