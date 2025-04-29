"""
This file will contain the classes for the neural nets  
    
"""
import torch
import torch.nn as nn

# conv feature extractor
# want the same features to be extracted for the actor and the critic
# so use the same feature extractor for both a & c

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            # First conv layer: input channels = 4 (frame stack), output = 32
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Second conv layer: 32 -> 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Third conv layer: 64 -> 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            # Fourth conv layer: 128 -> 128 filters
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        # Pass input through the conv layers
        return self.conv_layers(x)

# actor network outputs logits over action space (= 18 actions)
class Actor(nn.Module):
    def __init__(self, n_actions):
        """
        Outputs logits over 18 discrete actions.
        """
        super(Actor, self).__init__()
        self.features = CNNFeatureExtractor()
        self.fc = nn.Sequential(
            nn.Flatten(),
            # Fully connected layer to interpret features
            nn.Linear(128 * 5 * 5, 512), # that final output of the CNN is 128 feature maps of size 5x5
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout added - avoids overfitting
            # output logits for each action (18 actions)
            nn.Linear(512, n_actions)
        )
        self.optimizer = torch.optim.Adam

    def forward(self, x):
        # Forward pass through feature extractor then fully connected head
        x = self.features(x)
        return self.fc(x)
    
    def compile (self):
        pass
    
    def train (self):
        pass

# critic network outputs scalar value for state
class Critic(nn.Module):
    def __init__(self):
        """
        outputs a value estimate for a given state
        """
        super(Critic, self).__init__()
        self.features = CNNFeatureExtractor()
        self.fc = nn.Sequential(
            nn.Flatten(),
            # FC layer
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Final output layer - outputs a scalar value V(s)
            nn.Linear(512, 1)
        )
        self.optimizer = torch.optim.Adam

    def forward(self, x):
        # Forward pass through feature extractor then FC head
        x = self.features(x)
        return self.fc(x)
    
    def compile (self):
        pass
    
    def train (self):
        pass
    
model = Critic()
print(model)