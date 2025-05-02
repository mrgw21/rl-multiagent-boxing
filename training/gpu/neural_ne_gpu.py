import torch
import torch.nn as nn
import torch.nn.init as init
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__all__ = ["device", "Actor", "Critic"]

# Shared CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((5, 5))  # Output shape fixed to (128, 5, 5)
        )
        self._init_weights()

    def _init_weights(self):
        """orthogonal initialization of the weights for conv layers, bias is set to 0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv_layers(x)

# --------- Actor Network ---------
class Actor(nn.Module):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.features = CNNFeatureExtractor()
        
        # FC layers are nowexplicitly defined, not in sequential
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.policy_logits = nn.Linear(128, n_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)  # Based on ICLR 

        self._init_weights()

    def _init_weights(self):
        # orthogonal init for hidden layers
        init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        init.orthogonal_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        
        # init for policy output logits with 0.01 gain as recommended in ICLR
        init.orthogonal_(self.policy_logits.weight, gain=0.01)

        # bias is set to 0
        for layer in [self.fc1, self.fc2, self.fc3, self.policy_logits]:
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        logits = self.policy_logits(x)
        return self.softmax(logits) 

    def save_model(self, path="training/models/actor_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, pathname="training/models/actor_model.pth"):
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
        
        # FC layers are nowexplicitly defined, not in sequential
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        
        self._init_weights()

    def _init_weights(self):
        # orthogonal init for hidden layers
        init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        init.orthogonal_(self.fc2.weight, gain=1.0)
        
        # bias is set to 0
        for layer in [self.fc1, self.fc2]:
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def save_model(self, path="training/models/critic_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, pathname="training/models/critic_model.pth"):
        if pathname is None or pathname == "None":
            return
        state_dict = torch.load(pathname, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)