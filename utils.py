import csv
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


def rotate_direction(direction: int, rotation: int):
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    index_direction = clock_wise.index(direction)

    if rotation == 1:  # clockwise
        next_idx = (index_direction + 1) % 4
    elif rotation == -1:  # counterclockwise
        next_idx = (index_direction - 1) % 4
    else:
        raise ValueError("Rotation must be '1' (clockwise) or '-1' (counterclockwise).")

    return clock_wise[next_idx]


def save_logs(log_file_name: str = "logs.csv", *args):
    with open(log_file_name, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(args)
