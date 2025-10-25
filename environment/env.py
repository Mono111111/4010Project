# env.py
# Simple Snake Game MVP with keyboard control
import pygame
import random
import sys
from environment import config  # import our settings

class SnakeGame:
    def __init__(self):
        # Initialize pygame
        pygame.init()

        # Create game window
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        pygame.display.set_caption("Snake MVP")

        # Clock for controlling speed
        self.clock = pygame.time.Clock()

        # Reset the game state
        self.reset()

    def reset(self):

        self.game_over = False

        # Snake starts in the middle
        self.snake = [[config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2]]
        self.direction = [0, -config.GRID_SIZE]  # start moving up

        # List to hold multiple foods
        self.foods = []
        self.spawn_initial_foods(config.FOOD_INIT_COUNT)

        self.obstacles = []
        self._spawn_obstacles(config.OBSTACLE_COUNT)
        
        # Score and energy
        self.score = 0
        self.energy = config.INITIAL_ENERGY

        # timing
        self.start_time = pygame.time.get_ticks()  # ms since pygame.init()
        self.survival_ms = None                    # filled at game over

        # Dynamic obstacles
        self._spawn_dynamic_obstacles(config.DYN_OBS_COUNT)

    def get_observation(self):
        # Return a simple, low-dim observation for RL
        hx, hy = self.snake[0]
        # choose nearest food (Manhattan distance))
        fx, fy, _ = min(self.foods, key=lambda f: abs(f[0]-hx)+abs(f[1]-hy))
        dx, dy = fx - hx, fy - hy
        # simple observation: head pos, energy, score, food delta
        return (hx, hy, self.energy, self.score, dx, dy)

    def _random_food_cell(self):
        while True:
            x = random.randrange(0, config.WINDOW_WIDTH, config.GRID_SIZE)
            y = random.randrange(0, config.WINDOW_HEIGHT, config.GRID_SIZE)
            # not on snake or other foods
            if [x, y] not in self.snake and all(f[0] != x or f[1] != y for f in self.foods):
                return x, y

    def spawn_one_food(self):
        x, y = self._random_food_cell()
        ftype = "mystery" if random.random() < config.MYSTERY_PROB else "normal"
        return [x, y, ftype]

    def spawn_initial_foods(self, n):
        self.foods = [self.spawn_one_food() for _ in range(n)]

    def _random_empty_cell(self):
        # returns a random cell not occupied by snake, food, or obstacles
        while True:
            x = random.randrange(0, config.WINDOW_WIDTH, config.GRID_SIZE)
            y = random.randrange(0, config.WINDOW_HEIGHT, config.GRID_SIZE)
            occ_food = any(f[0] == x and f[1] == y for f in getattr(self, 'foods', []))
            occ_obst = any(o[0] == x and o[1] == y for o in getattr(self, 'obstacles', []))
            if [x, y] not in self.snake and not occ_food and not occ_obst:
                return x, y

    def _spawn_obstacles(self, n):
        self.obstacles = []
        for _ in range(n):
            x, y = self._random_empty_cell()
            self.obstacles.append([x, y])

    def _spawn_dynamic_obstacles(self, n):
        self.dynamic_obstacles = []
        for _ in range(n):
            x, y = self._random_empty_cell()
            # ramdom initial velocity
            if random.random() < 0.5:
                vel = [config.GRID_SIZE, 0]
            else:
                vel = [0, config.GRID_SIZE]
            self.dynamic_obstacles.append([x, y, vel])

    def _update_dynamic_obstacles(self):
        updated = []
        for x, y, vel in self.dynamic_obstacles:
            nx = (x + vel[0]) % config.WINDOW_WIDTH
            ny = (y + vel[1]) % config.WINDOW_HEIGHT
            # simple collision check, reverse if collides
            coll_snake = [nx, ny] in self.snake
            coll_static = any(nx == ox and ny == oy for ox, oy in self.obstacles)
            if coll_snake or coll_static:
                vel = [-vel[0], -vel[1]]
                nx = (x + vel[0]) % config.WINDOW_WIDTH
                ny = (y + vel[1]) % config.WINDOW_HEIGHT
            updated.append([nx, ny, vel])
        self.dynamic_obstacles = updated

    def step(self):
        old_score = self.score

        # update dynamic obstacles
        self._update_dynamic_obstacles()

        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != [0, config.GRID_SIZE]:
                    self.direction = [0, -config.GRID_SIZE]
                elif event.key == pygame.K_DOWN and self.direction != [0, -config.GRID_SIZE]:
                    self.direction = [0, config.GRID_SIZE]
                elif event.key == pygame.K_LEFT and self.direction != [config.GRID_SIZE, 0]:
                    self.direction = [-config.GRID_SIZE, 0]
                elif event.key == pygame.K_RIGHT and self.direction != [-config.GRID_SIZE, 0]:
                    self.direction = [config.GRID_SIZE, 0]

        # move
        new_x = (self.snake[0][0] + self.direction[0]) % config.WINDOW_WIDTH
        new_y = (self.snake[0][1] + self.direction[1]) % config.WINDOW_HEIGHT
        head = [new_x, new_y]
        self.snake.insert(0, head)

        # check if ate food
        ate_idx = -1
        for i, f in enumerate(self.foods):
            if head[0] == f[0] and head[1] == f[1]:
                ate_idx = i
                if f[2] == "normal":
                    self.score += config.SCORE_GAIN_NORMAL
                    self.energy = min(config.INITIAL_ENERGY, self.energy + config.ENERGY_GAIN_NORMAL)
                    print(f"[EVENT] Ate normal food: +{config.SCORE_GAIN_NORMAL} score, +{config.ENERGY_GAIN_NORMAL} energy")
                else:  # mystery
                    outcome = random.random()
                    if outcome < 0.05:
                        self.score += 50; print("[EVENT] Mystery -> Jackpot! +50 score")
                    elif outcome < 0.20: # 0.05-0.20
                        self.score += 10; print("[EVENT] Mystery -> Bonus! +10 score")
                    elif outcome < 0.30: # 0.20-0.30
                        self.score -= 10; print("[EVENT] Mystery -> Ouch! -10 score")
                    elif outcome < 0.50: # 0.30-0.50
                        nx = random.randrange(0, config.WINDOW_WIDTH, config.GRID_SIZE)
                        ny = random.randrange(0, config.WINDOW_HEIGHT, config.GRID_SIZE)
                        self.snake[0] = [nx, ny]
                        head = self.snake[0]
                        print("[EVENT] Mystery -> Teleport!")
                    else: # 0.50-1.00
                        self.score -= 5 
                        print("[EVENT] Mystery -> Penalty! -5 score")
                break

        # if ate, spawn new food, else pop tail
        if ate_idx >= 0:
            self.foods[ate_idx] = self.spawn_one_food()
        else:
            self.snake.pop()

        # dynamic obstacle check
        if any(head[0] == dx and head[1] == dy for dx, dy, _ in self.dynamic_obstacles):
            self.score -= config.DYN_OBS_PENALTY_SCORE
            print(f"[EVENT] Hit dynamic obstacle: -{config.DYN_OBS_PENALTY_SCORE} score")

        # obstacle check
        if any(head[0] == ox and head[1] == oy for ox, oy in self.obstacles):
            self.energy -= config.OBSTACLE_PENALTY_ENERGY
            print(f"[EVENT] Hit obstacle: -{config.OBSTACLE_PENALTY_ENERGY} energy")
            if self.energy <= 0:
                self.energy = 0
                self._end_game()

        # self-bite check
        if head in self.snake[1:]:
            self.energy -= 10
            print("[EVENT] Self-bite: -10 energy")
            if self.energy <= 0:
                self.energy = 0 
                self._end_game()

        # step energy loss
        self.energy -= config.ENERGY_LOSS_PER_STEP
        if self.energy <= 0:
            self.energy = 0
            self._end_game()
        if self.score >= 100:
            self._end_game()

        self.last_reward = self.score - old_score  # temporary reward


    def render(self):
        self.screen.fill(config.COLOR_BACKGROUND)

        # draw snake
        for pos in self.snake:
            pygame.draw.rect(self.screen, config.COLOR_SNAKE,
                             pygame.Rect(pos[0], pos[1], config.GRID_SIZE, config.GRID_SIZE))

        # draw foods
        for fx, fy, ft in self.foods:
            color = config.COLOR_FOOD if ft == "normal" else (255, 255, 0)
            pygame.draw.rect(self.screen, color, pygame.Rect(fx, fy, config.GRID_SIZE, config.GRID_SIZE))

        # draw obstacles
        for ox, oy in self.obstacles:
            pygame.draw.rect(
                self.screen, config.COLOR_OBSTACLE,
                pygame.Rect(ox, oy, config.GRID_SIZE, config.GRID_SIZE)
            )

        # draw dynamic obstacles
        for dx, dy, _ in self.dynamic_obstacles:
            pygame.draw.rect(
                self.screen, config.COLOR_DYN_OBS,
                pygame.Rect(dx, dy, config.GRID_SIZE, config.GRID_SIZE)
            )


        # HUD
        font = pygame.font.SysFont(None, 30)

        if self.survival_ms is not None:
            elapsed_s = self.survival_ms / 1000.0
        else:
            elapsed_s = (pygame.time.get_ticks() - self.start_time) / 1000.0

        hud = font.render(
            f"Score: {self.score}  Energy: {self.energy}  Time: {elapsed_s:.1f}s",
            True, config.COLOR_TEXT
        )
        self.screen.blit(hud, (10, 10))

        # Game Over
        if self.game_over:
            self._draw_center_text("GAME OVER", 48, config.COLOR_TEXT, y_offset=-30)
            self._draw_center_text(f"Final Score: {self.score}", 32, config.COLOR_TEXT, y_offset=10)
            self._draw_center_text(f"Survival: {elapsed_s:.1f}s", 32, config.COLOR_TEXT, y_offset=45)

        pygame.display.flip()

    
    def _draw_center_text(self, text, size, color, y_offset=0):
        font = pygame.font.SysFont(None, size)
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(config.WINDOW_WIDTH//2,
                                     config.WINDOW_HEIGHT//2 + y_offset))
        self.screen.blit(surf, rect)

    def run(self):
        while True:
            self.step()
            self.render()
            if self.game_over:
                print("[GAME OVER] Final Score:", self.score)
                if self.survival_ms is not None:
                    print("[GAME OVER] Survival:", f"{self.survival_ms/1000:.1f}s")
                self._show_game_over(2000)
                break
            self.clock.tick(config.FPS)

    def _end_game(self):
        if not self.game_over:
            self.game_over = True
            self.survival_ms = pygame.time.get_ticks() - self.start_time

    def _show_game_over(self, ms=2000):
        # show game over screen for ms milliseconds
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # keep rendering the game over screen
            self.render()
            self.clock.tick(30)


