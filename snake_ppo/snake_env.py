import random
import numpy as np
import pygame
from config import *


class Snake:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.length = 3
        cx = (SCREEN_WIDTH // 2 // BLOCK_SIZE) * BLOCK_SIZE
        cy = (SCREEN_HEIGHT // 2 // BLOCK_SIZE) * BLOCK_SIZE
        self.positions = [(cx, cy), (cx - BLOCK_SIZE, cy), (cx - 2 * BLOCK_SIZE, cy)]
        self.direction = (1, 0)
        self.is_alive = True
    
    def move(self):
        if not self.is_alive:
            return
        head = self.positions[0]
        new_head = (head[0] + self.direction[0] * BLOCK_SIZE, head[1] + self.direction[1] * BLOCK_SIZE)
        
        if (new_head[0] < 0 or new_head[0] >= SCREEN_WIDTH or
            new_head[1] < 0 or new_head[1] >= SCREEN_HEIGHT or
            new_head in self.positions[:-1]):
            self.is_alive = False
            return
        
        self.positions.insert(0, new_head)
        if len(self.positions) > self.length:
            self.positions.pop()
    
    def turn(self, direction):
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction
    
    def grow(self):
        self.length += 1
    
    def draw(self, surface):
        for i, pos in enumerate(self.positions):
            rect = pygame.Rect(pos[0], pos[1], BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, DARK_GREEN if i == 0 else GREEN, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class Food:
    def __init__(self, snake_positions=None):
        self.respawn(snake_positions)
    
    def respawn(self, snake_positions=None):
        while True:
            self.position = (
                random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE),
                random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE)
            )
            if snake_positions is None or self.position not in snake_positions:
                break
    
    def draw(self, surface):
        rect = pygame.Rect(self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(surface, RED, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)


class SnakeEnv:
    DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.state_dim = 25
        self.action_dim = 4
        self.screen = self.clock = self.font = None
        self.snake = self.food = None
        self.score = self.steps = 0
        self.max_steps = 1000
        
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake PPO")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)
    
    def reset(self):
        self.snake = Snake()
        self.food = Food(self.snake.positions)
        self.score = self.steps = self.steps_since_food = 0
        return self._get_state()
    
    def _get_state(self):
        head = self.snake.positions[0]
        food = self.food.position
        d = self.snake.direction
        
        def is_danger(pos):
            return (pos[0] < 0 or pos[0] >= SCREEN_WIDTH or
                    pos[1] < 0 or pos[1] >= SCREEN_HEIGHT or
                    pos in self.snake.positions[1:])
        
        def ray_distance(dx, dy, max_dist=10):
            for i in range(1, max_dist + 1):
                pos = (head[0] + dx * BLOCK_SIZE * i, head[1] + dy * BLOCK_SIZE * i)
                if is_danger(pos):
                    return i / max_dist
            return 1.0
        
        dangers = [is_danger((head[0] + dx * BLOCK_SIZE, head[1] + dy * BLOCK_SIZE)) 
                   for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        
        ray_dists = [ray_distance(dx, dy) for dx, dy in 
                     [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]]
        
        food_dir = [food[0] < head[0], food[0] > head[0], food[1] < head[1], food[1] > head[1]]
        food_dist = [abs(food[0] - head[0]) / SCREEN_WIDTH, abs(food[1] - head[1]) / SCREEN_HEIGHT]
        
        cur_dir = [d == (0, -1), d == (0, 1), d == (-1, 0), d == (1, 0)]
        
        length_norm = [min(self.snake.length / 50, 1.0)]
        head_norm = [head[0] / SCREEN_WIDTH, head[1] / SCREEN_HEIGHT]
        
        return np.array([*dangers, *ray_dists, *food_dir, *food_dist, *cur_dir, *length_norm, *head_norm], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        self.steps_since_food += 1
        old_dist = abs(self.snake.positions[0][0] - self.food.position[0]) + abs(self.snake.positions[0][1] - self.food.position[1])
        
        self.snake.turn(self.DIRECTIONS[action])
        self.snake.move()
        
        if not self.snake.is_alive:
            return self._get_state(), -10.0, True, {'score': self.score}
        
        done = False
        if self.snake.positions[0] == self.food.position:
            self.snake.grow()
            self.score += 1
            self.food.respawn(self.snake.positions)
            self.steps_since_food = 0
            reward = 10.0 + self.snake.length * 0.5
        else:
            new_dist = abs(self.snake.positions[0][0] - self.food.position[0]) + abs(self.snake.positions[0][1] - self.food.position[1])
            reward = 0.1 if new_dist < old_dist else -0.15
            
            if self.steps_since_food > 100:
                reward -= 0.1
        
        if self.steps >= self.max_steps:
            done, reward = True, -5.0
        
        return self._get_state(), reward, done, {'score': self.score}
    
    def render(self, fps=None):
        if self.render_mode != 'human':
            return True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        self.screen.fill(BLACK)
        for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (SCREEN_WIDTH, y))
        
        self.snake.draw(self.screen)
        self.food.draw(self.screen)
        self.screen.blit(self.font.render(f"Score: {self.score}  Length: {self.snake.length}", True, WHITE), (10, 10))
        pygame.display.flip()
        
        if fps:
            self.clock.tick(fps)
        return True
    
    def close(self):
        if self.screen:
            pygame.quit()
