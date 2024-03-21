import torch
import torch.nn as nn
import torch.optim as optim
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class Net(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        model_file_path: str = None,
    ):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=0),
        )
        if model_file_path:
            self.load(model_file_path)

    def forward(self, x):
        return self.linear_relu_stack(x)

    def save(self, file_name: str = "model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_path: str):
        if os.path.isfile(file_path):
            self.load_state_dict(torch.load(file_path))
            print("Model loaded.")
        else:
            print("Model file not found.")


class QLearning:
    def __init__(self, model: Net, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self, state: list, action: list, reward: int, next_state: list, game_over: bool
    ):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        prediction = self.model(state)

        with torch.no_grad():
            target = prediction.clone()
            for idx, go in enumerate(game_over):
                if not go:
                    Q_new = reward[idx] + self.gamma * torch.max(
                        self.model(next_state[idx])
                    )
                else:
                    Q_new = reward[idx]

                target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
