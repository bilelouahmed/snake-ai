from agent import *
from plot import plot

def inference(agent:Agent):
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

            print(f"Game : {agent.game_iteration} - Score : {score} - Record : {record}")


if __name__ == "__main__":
    agent = Agent(model_file_path=None)
    inference(agent)