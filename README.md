# Snake Resolution by Reinforcement Learning (Q-Learning)

## Setup and Run

To do...

## Decisions

Actions :
- Straight
- Go left
- Go right

State :
- Danger :
    - Left
    - Right
    - Straight
    - Straight Left
    - Straight Right
    - Behind Left
    - Behind Right
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

Reward : 
- 1 if food has been eaten
- -1 if game over : Classic Game Over or Time > 100 * Length of the Snake
- else 0

# Model

- Loss : Bellman Equation
- Epsilon : max(0.90 - 0.02*iteration_number)