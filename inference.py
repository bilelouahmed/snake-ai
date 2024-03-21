from agent import *
import argparse
from plot import plot


def inference(agent: Agent):
    game_instance = Snake(display_game=True)

    record = 0

    scores, running_mean_scores = [], []

    while True:
        state_old = agent.get_state(game_instance)

        final_move = agent.predict_movement(state_old, disable_randomness=True)

        reward, game_over, score = game_instance.play_step(final_move)
        agent.get_state(game_instance)

        if game_over:
            game_instance.reset_game()

            if score > record:
                record = score

            scores.append(score)
            running_mean_scores.append(np.mean(scores[-100:]))
            plot(agent.gamma, scores, running_mean_scores)

            print(
                f"Game : {agent.game_iteration} - Score : {score} - Record : {record}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Snake AI model."
    )
    parser.add_argument(
        "--model_file_path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    args = parser.parse_args()

    agent = Agent(model_file_path=args.model_file_path)
    inference(agent)
