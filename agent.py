import numpy as np
from game import Snake, Direction, Point
from collections import deque
import random
import torch
from model import *
import threading

MAX_MEMORY = 10000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    def __init__(self, epsilon:float=0.5, gamma:float=0.3, model_hidden_layers:int=128, model_file_name:str=None):
        self.epsilon = epsilon
        self.gamma = gamma # Discount Rate (potential future reward implied by decision)
        
        self.game_iteration = 1
        self.memory = deque(maxlen=MAX_MEMORY)
        
        self.model = Net(11, model_hidden_layers, 3, model_file_name).to(device)
        self.training = QLearning(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game:Snake):
        head = game.snake[0]
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.check_collision(point_r)) or 
            (dir_l and game.check_collision(point_l)) or 
            (dir_u and game.check_collision(point_u)) or 
            (dir_d and game.check_collision(point_d)),

            # Danger right
            (dir_u and game.check_collision(point_r)) or 
            (dir_d and game.check_collision(point_l)) or 
            (dir_l and game.check_collision(point_u)) or 
            (dir_r and game.check_collision(point_d)),

            # Danger left
            (dir_d and game.check_collision(point_r)) or 
            (dir_u and game.check_collision(point_l)) or 
            (dir_r and game.check_collision(point_u)) or 
            (dir_l and game.check_collision(point_d)),

            # Danger straight left

            # Danger straight right

            # Danger right right

            # Danger right left

            # Danger left right

            # Danger left left
            
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

    def predict_movement(self, state):
        move_predicted = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            move_predicted[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            move_predicted[move] = 1
        
        return move_predicted
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.training.train_step(state, action, reward, next_state, game_over)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.training.train_step(states, actions, rewards, next_states, game_overs)

    def store(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def increment_game_iteration(self):
        self.game_iteration += 1

    def update_epsilon(self, decrement:float=2.0*10**-4, sup_epsilon:float=0.05):
        self.epsilon = max(sup_epsilon, self.epsilon - decrement)


def train(display_plots=True):
    record = 0
    agent = Agent(epsilon=0.4, gamma=0.9)

    if not display_plots:
        from utils import plot

        scores, running_mean_scores, epsilons = [], [], []
        game_instance = Snake()

    else:
        game_instance = Snake(display_game=False)

    while True:
        state_old = agent.get_state(game_instance)

        final_move = agent.predict_movement(state_old)

        reward, game_over, score = game_instance.play_step(final_move)
        state_new = agent.get_state(game_instance)

        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        agent.store(state_old, final_move, reward, state_new, game_over)

        if game_over:
            game_instance.reset_game()
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(filename=f"record_{score}")

            if not display_plots:

                scores.append(score)
                running_mean_scores.append(np.mean(scores[-100:]))
                epsilons.append(agent.epsilon)
                plot(agent.gamma, scores, running_mean_scores, epsilons)

            print(f"Game : {agent.game_iteration} - Score : {score} - Record : {record}")

            agent.increment_game_iteration()
            agent.update_epsilon()


def multi_training(agent_epsilon:int=0.4, agent_gamma:int=0.9, epochs:int=500, workers:int=5, best_games_ratio:float=0.25):
    agent = Agent(epsilon=agent_epsilon, gamma=agent_gamma)
    lock = threading.Lock()

    def games_repetition(max_submemory:int=int(MAX_MEMORY/workers)):
        nonlocal main_games

        games = deque(max_submemory)
        game_instance = Snake(display_game=False)
        hits = []

        while len(games) < 0.9 * (max_submemory): # à revoir peut-être, peut-être à remplacer par BATCH_SIZE
            state_old = agent.get_state(game_instance)

            move = agent.predict_movement(state_old)

            reward, game_over, score = game_instance.play_step(move)
            state_new = agent.get_state(game_instance)

            hits.append((state_old, move, reward, state_new, game_over))

            if game_over:
                for state_old, move, reward, state_new, game_over in hits:
                    games.append((score, state_old, move, reward, state_new, game_over))

                hits = []
                
                game_instance.reset_game()

        print("Thread finished : length", len(games))

        with lock:
            main_games.extend(games)


    for epoch in range(epochs):
        main_games = []

        threads = []
        for _ in range(workers):
            t = threading.Thread(target=games_repetition, args=None)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        main_games.sort(key=lambda x: x[0], reverse=True)
        top_games = main_games[:int(best_games_ratio*len(main_games))]
        print("Length Main Games", len(main_games))
        print("Length Top Games", len(top_games))
        print("Top Games", top_games)

        

        # faire un entraînement et réitérer

        #agent.increment_game_iteration()
        #agent.update_epsilon()


if __name__ == '__main__':
    multi_training(agent_epsilon=0.4, agent_gamma=0.9, epochs=1, workers=2, best_games_ratio=0.01)