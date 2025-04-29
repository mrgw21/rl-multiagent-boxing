import torch
import torch.nn as nn
import os

# ---- Device setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Force GPU 2
__all__ = ["device", "Actor", "Critic"]

# --------- Shared CNN Feature Extractor ---------
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((5, 5))  # Output shape fixed to (128, 5, 5)
        )

    def forward(self, x):
        return self.conv_layers(x)

# --------- Actor Network ---------
class Actor(nn.Module):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.features = CNNFeatureExtractor()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

    def save_model(self, path="training/models/actor_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)

    def load_model(self, pathname="training/models/actor_model.pth"):
        os.makedirs(os.path.dirname(pathname), exist_ok=True) 
        if pathname is None or pathname == "None":
            return
        state_dict = torch.load(pathname, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)

# --------- Critic Network ---------
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.features = CNNFeatureExtractor()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

    def save_model(self, path="training/models/critic_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        torch.save(self.state_dict(), path)

    def load_model(self, pathname="training/models/critic_model.pth"):
        if pathname is None or pathname == "None":
            return
        state_dict = torch.load(pathname, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)
