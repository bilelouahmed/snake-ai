from utils import *
import pygame
import random
from collections import namedtuple
import numpy as np

SCORE_FONT_SIZE = 25
GAME_OVER_FONT_SIZE = 20
BLOCK_SIZE = 20
GAME_SPEED = 10
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RED = (220, 20, 60)
GREEN = (50,205,50)
BLUE = (116, 208, 241)

DEFAULT_REWARDS = {
    "GAME_OVER": -100,
    "FOOD": 10,
    "NONE": 0
}

Point = namedtuple('Point', 'x, y')

class Snake():
    def __init__(self, width:int=600, height:int=400, rewards:dict=DEFAULT_REWARDS, display_game=True):
        self.width, self.height = width, height
        self.display_game = display_game

        if self.display_game:
            pygame.init()
            self.game = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')

            self.clock = pygame.time.Clock()

            self.game_close_style = pygame.font.SysFont("Courier New", GAME_OVER_FONT_SIZE, bold=True)
            self.score_font = pygame.font.SysFont("Courier New", SCORE_FONT_SIZE)

            self.game_speed = GAME_SPEED

        self.block_size = BLOCK_SIZE
        
        self.rewards = rewards

        self.reset_game()

    def generate_food(self):
        possible_positions = []

        for x in range(0, self.width - self.block_size, self.block_size):
            for y in range(0, self.height - self.block_size, self.block_size):
                if Point(x, y) not in self.snake:
                    possible_positions.append(Point(x, y))
        
        self.food = random.choice(possible_positions)

    def move(self, action):
        x, y = self.head.x, self.head.y

        if np.array_equal(action, [0, 1, 0]):  # Turn right
            self.direction = rotate_direction(self.direction, 1)
        else:  # Turn left
            self.direction = rotate_direction(self.direction, -1)

        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

    def check_collision(self, head:Point=None):
        if head == None:
            head = self.head
        if head.x >= self.width or head.x < 0 or head.y >= self.height or head.y < 0:
            return True
        if head in self.snake[:-1]:
            return True
        return False

    def play_step(self, action):
        self.iteration += 1

        self.move(action)

        reward = self.rewards["NONE"]

        if self.check_collision() or self.iteration > 100*self.snake_length:
            reward = self.rewards["GAME_OVER"]
            self.game_over = True
            return reward, self.game_over, self.get_score()

        if self.head.x == self.food.x and self.head.y == self.food.y:
            reward = self.rewards["FOOD"]
            self.generate_food()
            self.snake_length += 1

        self.snake.append(self.head)
        if len(self.snake) > self.snake_length:
            del self.snake[0]

        if self.display_game:
            self.routine()

        return reward, self.game_over, self.get_score()

    def routine(self):
        self.game.fill(BLUE)
        self.draw_snake(self.snake)
        self.print_score()
        pygame.draw.rect(self.game, RED, [self.food.x, self.food.y, self.block_size, self.block_size])
        pygame.display.update()

        self.clock.tick(self.game_speed)

    def reset_game(self):
        self.game_over = False
        self.iteration = 0

        self.head = Point(self.width / 2, self.height / 2)
        self.direction = Direction.RIGHT

        self.snake = [self.head]
        self.snake_length = 1

        self.generate_food()

    def get_score(self):
        return self.snake_length - 1

    def print_score(self):
        value = self.score_font.render("Score : " + str(self.get_score()), True, BLACK)
        self.game.blit(value, [0, 0])

    def draw_snake(self, snake):
        for x in snake:
            pygame.draw.rect(self.game, BLACK, [x[0], x[1], self.block_size, self.block_size])