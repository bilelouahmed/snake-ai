from game import *
from model import *
from collections import deque
import random

MAX_MEMORY = 100000
BATCH_SIZE = 50
LR = 0.001

class Agent():
    def __init__(self, epsilon:float=0.5, gamma:float=0.3, model_hidden_layers:int=256, model_file_path:str=None):
        self.epsilon = epsilon
        self.gamma = gamma # Discount Rate (potential future reward implied by decision)
        
        self.game_iteration = 1
        self.memory = []
        
        self.model = Net(16, model_hidden_layers, 3, model_file_path).to(device)
        self.training = QLearning(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game:Snake):
        head = game.head

        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)
        point_ll = Point(head.x - 2*game.block_size, head.y)
        point_rr = Point (head.x + 2*game.block_size, head.y)
        point_uu = Point(head.x, head.y - 2*game.block_size)
        point_dd = Point(head.x, head.y + 2*game.block_size)
        point_lu = Point(head.x - game.block_size, head.y - game.block_size)
        point_ld = Point(head.x - game.block_size, head.y + game.block_size)
        point_ru = Point(head.x + game.block_size, head.y - game.block_size)
        point_rd = Point(head.x + game.block_size, head.y + game.block_size)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_u and game.check_collision(point_u)) or
            (dir_r and game.check_collision(point_r)) or 
            (dir_d and game.check_collision(point_d)) or
            (dir_l and game.check_collision(point_l)),

            # Danger double straight
            (dir_u and game.check_collision(point_uu))
            (dir_r and game.check_collision(point_rr)) or
            (dir_d and game.check_collision(point_dd)) or
            (dir_l and game.check_collision(point_ll)),

            # Danger right
            (dir_u and game.check_collision(point_r)) or 
            (dir_r and game.check_collision(point_d)) or
            (dir_d and game.check_collision(point_l)) or 
            (dir_l and game.check_collision(point_u)),

            # Danger left
            (dir_u and game.check_collision(point_l)) or
            (dir_r and game.check_collision(point_u)) or 
            (dir_d and game.check_collision(point_r)) or 
            (dir_l and game.check_collision(point_d)),

            # Danger behind left
            (dir_u and game.check_collision(point_ld)) or
            (dir_r and game.check_collision(point_lu)) or
            (dir_d and game.check_collision(point_ru)) or
            (dir_l and game.check_collision(point_rd)),

            # Danger behind right
            (dir_u and game.check_collision(point_rd)) or
            (dir_r and game.check_collision(point_ld)) or
            (dir_d and game.check_collision(point_lu)) or
            (dir_l and game.check_collision(point_ru)),

            # Danger ahead left
            (dir_u and game.check_collision(point_lu)) or
            (dir_r and game.check_collision(point_ru)) or
            (dir_d and game.check_collision(point_rd)) or
            (dir_l and game.check_collision(point_ld)),

            # Danger ahead right
            (dir_u and game.check_collision(point_ru)) or
            (dir_r and game.check_collision(point_rd)) or
            (dir_d and game.check_collision(point_ld)) or
            (dir_l and game.check_collision(point_lu)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y  # Food down
            ]

        return np.array(state, dtype=int)

    def predict_movement(self, state, disable_randomness:bool=False):
        move_predicted = [0, 0, 0]
        if not disable_randomness:
            if random.random() < self.epsilon:
                move = random.randint(0, 2)
                move_predicted[move] = 1

                return move_predicted
            
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        move_predicted[move] = 1

        return move_predicted
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.training.train_step(state, action, reward, next_state, game_over)

    def train_long_memory(self):
        random.shuffle(self.memory)
        
        for start in range(0, len(self.memory), BATCH_SIZE):
            mini_sample = self.memory[start:start + BATCH_SIZE]

            scores, states, actions, rewards, next_states, game_overs = zip(*mini_sample)
            self.training.train_step(states, actions, rewards, next_states, game_overs)

    def store(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def increment_game_iteration(self):
        self.game_iteration += 1

    def update_epsilon(self, decrement:float=2.0*10**-4, sup_epsilon:float=0.05):
        self.epsilon = max(sup_epsilon, self.epsilon - decrement)

    def clear_memory(self):
        self.memory = []