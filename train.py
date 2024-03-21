import argparse
from agent import *
import threading
from utils import save_logs

MAX_MEMORY = 100000
DEFAULT_WORKERS = 2
DEFAULT_EPOCHS = 300
DEFAULT_BEST_GAMES_RATIO = 0.25


def unithread_training(agent: Agent, display_plots: bool = True):
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

            print(
                f"Game : {agent.game_iteration} - Score : {score} - Record : {record}"
            )

            agent.increment_game_iteration()
            agent.update_epsilon()


def multi_training(
    agent: Agent,
    epochs: int = DEFAULT_EPOCHS,
    workers: int = DEFAULT_WORKERS,
    best_games_ratio: float = DEFAULT_BEST_GAMES_RATIO,
):
    lock = threading.Lock()

    def games_repetition(max_submemory: int = int(MAX_MEMORY / workers)):
        games = []
        game_instance = Snake(display_game=False)
        hits = []

        while len(games) < max_submemory:
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

    def game_test(epoch: int, n_tests: int = 1000):
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
            agent.model.save(filename=f"checkpoint_{record}")

    for epoch in range(1, epochs + 1):
        threads = []
        for _ in range(workers):
            t = threading.Thread(target=games_repetition)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        agent.memory = sorted(agent.memory, key=lambda x: x[0], reverse=True)[
            : int(best_games_ratio * len(agent.memory))
        ]

        agent.train_long_memory()

        game_test(epoch=epoch, n_tests=1000)

        agent.clear_memory()
        agent.update_epsilon(decrement=1.5 * 10**-3, inf_epsilon=0.1)

    agent.model.save(filename=f"trained_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Snake AI model.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Exploration rate (default: {DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"Discount factor (default: {DEFAULT_GAMMA})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate for training (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--best_games_ratio",
        type=float,
        default=DEFAULT_BEST_GAMES_RATIO,
        help=f"Ratio of best games to retain (default: {DEFAULT_BEST_GAMES_RATIO})",
    )
    parser.add_argument(
        "--max_memory",
        type=int,
        default=MAX_MEMORY,
        help=f"Maximum memory size (default: {MAX_MEMORY})",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=DEFAULT_HIDDEN_UNITS,
        help=f"Number of units on the hidden neural layer (default: {DEFAULT_HIDDEN_UNITS})",
    )
    parser.add_argument(
        "--model_file_path",
        type=str,
        default=None,
        help="Path to the model file to finetune (default: None, to not use with hidden_units argument.)",
    )
    args = parser.parse_args()

    agent = Agent(
        epsilon=args.epsilon,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_file_path=args.model_file_path,
        model_hidden_units=args.hidden_units,
    )
    multi_training(
        agent,
        epochs=args.epochs,
        workers=args.workers,
        best_games_ratio=args.best_games_ratio,
    )
