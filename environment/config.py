# config.py
# Basic configuration for Snake MVP environment

# Window (game screen) size
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Grid size (the snake and food move in blocks, like tiles)
GRID_SIZE = 20

# Snake settings
INITIAL_ENERGY = 500   # starting energy
ENERGY_LOSS_PER_STEP = 1  # energy consumed every move
ENERGY_GAIN_NORMAL = 5    # energy gained from normal food
SCORE_GAIN_NORMAL = 5     # score gained from normal food

# Food settings
FOOD_INIT_COUNT = 5
MYSTERY_PROB = 0.2  # probability of spawning mystery food

# Colors (RGB format)
COLOR_BACKGROUND = (0, 0, 0)       # black
COLOR_SNAKE = (0, 255, 0)          # green
COLOR_FOOD = (255, 0, 0)           # red
COLOR_TEXT = (255, 255, 255)       # white

# Game speed (frames per second)
FPS = 10
