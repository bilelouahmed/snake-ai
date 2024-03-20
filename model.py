import torch
import torch.nn as nn
import torch.optim as optim
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, filepath:str=None):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )
        if filepath:
            self.load(filepath)

    def forward(self, x):
        return self.linear_relu_stack(x)

    def save(self, filename='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        filename = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), filename)

    def load(self, filepath):
        if os.path.isfile(filepath):
            self.load_state_dict(torch.load(filepath))
            print("Model loaded.")
        else:
            print("Model file not found.")


class QLearning:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        prediction = self.model(state)

        with torch.no_grad():
            target = prediction.clone()
            for idx, go in enumerate(game_over):
                if not go:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                else:
                    Q_new = reward[idx]

                target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()