from agent import *
import threading
from utils import save_logs

def unithread_training(agent:Agent, display_plots=True):
    record = 0

    if not display_plots:
        from plot import plot

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


def multi_training(agent:Agent, epochs:int=500, workers:int=5, best_games_ratio:float=0.25):
    lock = threading.Lock()

    def games_repetition(max_submemory:int=int(MAX_MEMORY/workers)):
        games = deque(maxlen=max_submemory)
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

        with lock:
            agent.memory.extend(games)
    
    def game_test(epoch:int, n_tests:int=1000):
        scores = []
        record = 0

        game_instance = Snake(display_game=False)
        game_over = False
        
        for i in range(n_tests):
            while not game_over:
                state_old = agent.get_state(game_instance)
                move = agent.predict_movement(state_old, disable_randomness=True)
                reward, game_over, score = game_instance.play_step(move)
                agent.get_state(game_instance)

            scores.append(score)
            game_instance.reset_game()
            game_over = False

        avg_score = round(np.mean(scores), 3)
        std_score = round(np.std(scores), 3)

        print(f"Epoch {epoch} - Average score : {avg_score} +- {std_score}")
        save_logs("logs.csv", epoch, avg_score, std_score, agent.epsilon)

        if int(avg_score) > record:
            record = int(avg_score)
            agent.model.save(filename=f"model_record_{record}")

    for epoch in range(1, epochs + 1):
        threads = []
        for _ in range(workers):
            t = threading.Thread(target=games_repetition)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        agent.memory = sorted(agent.memory, key=lambda x: x[0], reverse=True)[:int(best_games_ratio * len(agent.memory))]

        agent.train_long_memory()

        game_test(epoch=epoch, n_tests=1000)

        agent.clear_memory()
        agent.update_epsilon(decrement=5*10**-3, sup_epsilon=0.1)

    agent.model.save(filename=f"trained_model")

if __name__ == "__main__":
    agent = Agent(epsilon=0.4, gamma=0.9, model_file_path=None)
    multi_training(agent, epochs=300, workers=3, best_games_ratio=0.25)