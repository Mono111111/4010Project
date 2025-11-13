# env.py
# Snake Game Environment
import pygame
import sys
from environment import config
from environment.core import SnakeGameCore
from environment.renderer import SnakeGameRenderer

class SnakeGame:
    
    def __init__(self):
        # Initialize pygame
        pygame.init()

        # Create game window
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption("Snake MVP")

        # Clock for controlling speed
        self.clock = pygame.time.Clock()

        # Initialize core logic and renderer
        self.core = SnakeGameCore()
        self.renderer = SnakeGameRenderer(self.screen)

    def reset(self):
        self.core.reset()

    def get_observation(self):
        return self.core.get_observation()

    @property
    def game_over(self):
        return self.core.game_over

    @property
    def score(self):
        return self.core.score

    @property
    def energy(self):
        return self.core.energy

    @property
    def last_reward(self):
        return self.core.last_reward

    @property
    def direction(self):
        return self.core.direction

    @direction.setter
    def direction(self, value):
        self.core.direction = value

    @property
    def food(self):
        return self.core.foods

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event):
        key_map = {
            pygame.K_UP: 'UP',
            pygame.K_DOWN: 'DOWN',
            pygame.K_LEFT: 'LEFT',
            pygame.K_RIGHT: 'RIGHT'
        }
        
        if event.key in key_map:
            self.core.handle_direction_input(key_map[event.key])

    def step(self):
        self.handle_events()
        self.core.step()

    def render(self):
        self.renderer.render(self.core)

    def run(self):
        while True:
            self.step()
            self.render()
            
            if self.game_over:
                print("[GAME OVER] Final Score:", self.score)
                if self.core.survival_ms is not None:
                    print("[GAME OVER] Survival:", f"{self.core.survival_ms/1000:.1f}s")
                self._show_game_over(2000)
                break
            
            self.clock.tick(config.FPS)

    def _show_game_over(self, ms=2000):
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.render()
            self.clock.tick(30)


if __name__ == "__main__":
    game = SnakeGame()
    game.run()