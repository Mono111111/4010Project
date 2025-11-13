# renderer.py
# Snake Game Renderer
import pygame
from environment import config

class SnakeGameRenderer:
    
    def __init__(self, screen):
        self.screen = screen
        
    def render(self, game_core):
        self.screen.fill(config.COLOR_BACKGROUND)
        
        self._draw_snake(game_core.snake)
        self._draw_foods(game_core.foods)
        self._draw_obstacles(game_core.obstacles)
        self._draw_dynamic_obstacles(game_core.dynamic_obstacles)
        
        self._draw_hud(game_core)
        
        self._draw_message(game_core)
        
        if game_core.game_over:
            self._draw_game_over(game_core)
        
        pygame.display.flip()
    
    def _draw_snake(self, snake):
        for pos in snake:
            pygame.draw.rect(
                self.screen, 
                config.COLOR_SNAKE,
                pygame.Rect(pos[0], pos[1], config.GRID_SIZE, config.GRID_SIZE)
            )
    
    def _draw_foods(self, foods):
        for fx, fy, ft in foods:
            color = config.COLOR_FOOD if ft == "normal" else (255, 255, 0)
            pygame.draw.rect(
                self.screen, 
                color, 
                pygame.Rect(fx, fy, config.GRID_SIZE, config.GRID_SIZE)
            )
    
    def _draw_obstacles(self, obstacles):
        for ox, oy in obstacles:
            pygame.draw.rect(
                self.screen, 
                config.COLOR_OBSTACLE,
                pygame.Rect(ox, oy, config.GRID_SIZE, config.GRID_SIZE)
            )
    
    def _draw_dynamic_obstacles(self, dynamic_obstacles):
        for dx, dy, _ in dynamic_obstacles:
            pygame.draw.rect(
                self.screen, 
                config.COLOR_DYN_OBS,
                pygame.Rect(dx, dy, config.GRID_SIZE, config.GRID_SIZE)
            )
    
    def _draw_hud(self, game_core):
        font = pygame.font.SysFont(None, 30)
        elapsed_s = game_core.get_elapsed_time()
        
        hud = font.render(
            f"Score: {game_core.score}  Energy: {game_core.energy}  Time: {elapsed_s:.1f}s",
            True, 
            config.COLOR_TEXT
        )
        self.screen.blit(hud, (10, 10))
    
    def _draw_message(self, game_core):
        now = pygame.time.get_ticks()
        if now < getattr(game_core, "message_start_ms", 0) and getattr(game_core, "message_msg", ""):
            smallfont = pygame.font.SysFont(None, 18)
            msg_surf = smallfont.render(game_core.message_msg, True, config.COLOR_TEXT)
            msg_rect = msg_surf.get_rect()
            msg_rect.bottomleft = (10, config.WINDOW_HEIGHT - 10)
            self.screen.blit(msg_surf, msg_rect)
    
    def _draw_game_over(self, game_core):
        elapsed_s = game_core.get_elapsed_time()
        
        self._draw_center_text("GAME OVER", 48, config.COLOR_TEXT, y_offset=-30)
        self._draw_center_text(f"Final Score: {game_core.score}", 32, config.COLOR_TEXT, y_offset=10)
        self._draw_center_text(f"Survival: {elapsed_s:.1f}s", 32, config.COLOR_TEXT, y_offset=45)
    
    def _draw_center_text(self, text, size, color, y_offset=0):
        font = pygame.font.SysFont(None, size)
        surf = font.render(text, True, color)
        rect = surf.get_rect(
            center=(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + y_offset)
        )
        self.screen.blit(surf, rect)