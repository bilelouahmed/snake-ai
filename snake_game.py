import pygame
import random
from collections import namedtuple
from enum import Enum

SCORE_FONT_SIZE = 25
GAME_OVER_FONT_SIZE = 20
BLOCK_SIZE = 20
GAME_SPEED = 5
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
    def __init__(self, width:int=600, height:int=400, rewards:dict=DEFAULT_REWARDS):
        pygame.init()
        self.width, self.height = width, height
        self.game_display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')

        self.clock = pygame.time.Clock()

        self.BLOCK_SIZE = BLOCK_SIZE
        self.GAME_SPEED = GAME_SPEED

        self.game_close_style = pygame.font.SysFont("Courier New", GAME_OVER_FONT_SIZE, bold=True)
        self.score_font = pygame.font.SysFont("Courier New", SCORE_FONT_SIZE)

        self.rewards = rewards
        self.direction = Direction.RIGHT

    def generate_food(self):
        x_food = round(random.randrange(0, self.width - self.BLOCK_SIZE) / self.BLOCK_SIZE) * self.BLOCK_SIZE
        y_food = round(random.randrange(0, self.height - self.BLOCK_SIZE) / self.BLOCK_SIZE) * self.BLOCK_SIZE
        self.food = Point(x_food, y_food)

    def move(self):
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = Point(x, y)

    def check_collision(self):
        if self.head.x >= self.width or self.head.x < 0 or self.head.y >= self.height or self.head.y < 0:
            return True
        for block in self.snake[:-1]:
            if block == self.head:
                return True
        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_close = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.game_close = True
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT

    def play_step(self):
        self.iteration += 1

        self.move()

        reward = self.rewards["NONE"]

        if self.check_collision() or self.iteration > 100 * len(self.snake):
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

        self.routine()

        return reward, self.game_over, self.get_score()

    def routine(self):
        self.game_display.fill(BLUE)
        self.draw_snake(self.snake)
        self.print_score()
        pygame.draw.rect(self.game_display, RED, [self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE])
        pygame.display.update()

        self.clock.tick(self.GAME_SPEED)

    def game_loop(self):
        self.game_close = False

        self.reset_game()

        while not self.game_close:
            self.handle_events()
            if not self.game_over:
                self.play_step()
            else:
                self.game_over_screen()

        pygame.quit()
        quit()

    def game_over_screen(self):
        self.game_display.fill(BLUE)
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

        self.snake = [self.head]
        self.snake_length = 1

        self.generate_food()

    def get_score(self):
        return self.snake_length

    def print_score(self):
        value = self.score_font.render("Score : " + str(self.snake_length), True, BLACK)
        self.game_display.blit(value, [0, 0])

    def print_game_over(self, msg, color):
        mesg = self.game_close_style.render(msg, True, color)
        text_width, text_height = self.game_close_style.size(msg)
        x = (self.width - text_width) / 2
        y = (self.height - text_height) / 2
        self.game_display.blit(mesg, (x, y))

    def draw_snake(self, snake):
        for block in snake:
            pygame.draw.rect(self.game_display, BLACK, [block.x, block.y, self.BLOCK_SIZE, self.BLOCK_SIZE])

if __name__ == "__main__":
    game = Snake()
    game.game_loop()