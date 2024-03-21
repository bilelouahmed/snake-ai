# Snake Resolution by Reinforcement Learning (Q-Learning)

A Snake game resolution using Reinforcement Learning, specifically the Q-Learning algorithm. The goal is to train an agent capable of playing the Snake game autonomously by learning a policy that maximizes its cumulative reward.

## Setup and Run

First, clone this repository to your local machine and navigate into the project directory.

```git clone https://github.com/bilelouahmed/snake-ai.git```

```cd snake-ai```

Make sure you have Python and pip installed. You can install the required libraries by running the following command :

```pip install -r requirements.txt```

If you want to train a model and try specific hyperparameters, run the training script using the following command :

```python train.py --epsilon 0.3 --epochs 500 --batch_size 1000 --learning_rate 0.0001 --workers 5 --best_games_ratio 0.3 --max_memory 200000 --model_file_path model_to_finetune.pth```

Once the model is trained, you can use the inference script to make predictions. Here's how to perform inference :

```python3 inference.py --model_file_path trained_model.pth```

## Decisions

**Actions :**
- Straight
- Go left
- Go right

**States :**
- Danger :
    - Left
    - Right
    - Straight
    - Double Straight
    - Behind Left
    - Behind Right
    - Ahead Left
    - Ahead Right
- Direction :
    - Left ?
    - Right ?
    - Up ?
    - Down ?
- Food Direction :
    - Left ?
    - Right ?
    - Up ?
    - Down ?

**Reward :**
- Food has been eaten : 10
- Game Over : -20

*Definition of Game Over :* Classic scenario where the snake hits the walls or itself, or when the time exceeds 100 times the length of the snake.

## Model

**Architecture :** This is a feedforward neural network implemented in PyTorch, consisting of one hidden layer with ReLU activation and a softmax output layer.

**Estimation of actions value :** Bellman equation. It is often written as:

$$V(s) = \max_a \left( R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right)$$

where:
- V(s) is the value of state s,
- R(s, a) is the immediate reward of taking action a in state s,
- P(s'|s, a) is the probability of transitioning to state s' by taking action a from state s,
- gamma is the discount factor representing the importance of future rewards.

**Criterion :** Mean Square Error

**Optimizer :** Adam

## Experience

| Parameter        |      Value      |
|------------------|:---------------:|
| Hidden units number    |       256       |
| Epsilon          |   0.4 - 1.5e-3 * epoch  |
| Gamma            |       0.9       |
| Batch size       |        50       |
| Learning rate    |       1e-3      |


