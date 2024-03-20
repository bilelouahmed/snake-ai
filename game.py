import pygame
import random
from collections import namedtuple
from enum import Enum
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
    "GAME_OVER": -10,
    "FOOD": 10,
    "NONE": 0
}

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

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
        x_food = round(random.randrange(0, self.width - self.block_size) / self.block_size) * self.block_size
        y_food = round(random.randrange(0, self.height - self.block_size) / self.block_size) * self.block_size
        self.food = Point(x_food, y_food)


    def move(self, action):
        x = self.head.x
        y = self.head.y

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index_direction = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Go straight
            new_dir = clock_wise[index_direction] 
        elif np.array_equal(action, [0, 1, 0]): # Turn right : RIGHT -> DOWN -> LEFT -> UP
            next_idx = (index_direction + 1) % 4
            new_dir = clock_wise[next_idx] 
        else:
            next_idx = (index_direction - 1) % 4
            new_dir = clock_wise[next_idx] # Turn left : RIGHT -> UP -> LEFT -> DOWN

        self.direction = new_dir

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


    # Useless for an agent -> to remove later
    def game_loop(self):
        self.game_close = False

        self.reset_game()

        while not self.game_close:
            if self.display_game:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_close = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.game_close = True
            if not self.game_over:
                self.play_step()
            else:
                self.reset_game()
        
        if self.display_game:
            pygame.quit()

        quit()


    # Useless for a bot -> to remove later
    def game_over_screen(self):
        self.game.fill(BLUE)
        self.print_game_over("Game Over ! Play again (CTRL+C) / Quit (CTRL+Q)", RED)
        self.print_score()
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.game_over = True
                    self.game_close = False
                elif event.key == pygame.K_c:
                    self.reset_game()


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


    def print_game_over(self, msg, color):
        mesg = self.game_close_style.render(msg, True, color)
        text_width, text_height = self.game_close_style.size(msg)
        x = (self.width - text_width) / 2
        y = (self.height - text_height) / 2
        self.game.blit(mesg, (x, y))


    def draw_snake(self, snake):
        for x in snake:
            pygame.draw.rect(self.game, BLACK, [x[0], x[1], self.block_size, self.block_size])