import torch
import torch.nn as nn
import torch.nn.init as init
import os

# ICLR link: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__all__ = ["device", "Actor", "Critic"]

# Shared CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        """
        Initialise the CNNFeatureExtractor.
        """
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), # 4 channels for the input image, 32 filters, kernel size 3, padding 1
            nn.ReLU(), # ReLU activation function
            nn.MaxPool2d(2), # Max pooling with a pool size of 2

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
        """
        Orthogonal initialisation of the weights for conv layers, bias is set to 0. 
        As per ICLR paper.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the CNNFeatureExtractor.
        """
        return self.conv_layers(x)

# --------- Actor Network ---------
class Actor(nn.Module):
    def __init__(self, n_actions):
        """
        Initialise the Actor.
        """
        super(Actor, self).__init__()  
        self.layers = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, padding=1), # 4 channels for the input image, 32 filters, kernel size 3, padding 1
        nn.ReLU(), # ReLU activation function
        nn.MaxPool2d(2), # Max pooling with a pool size of 2

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.AdaptiveAvgPool2d((5, 5)),  # Output shape fixed to (128, 5, 5)
        nn.Flatten(1),
        
        nn.Linear(128 * 5 * 5, 512), 
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions), 
        nn.Softmax(dim=-1) 
    )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)  # Based on ICLR paper

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialisation of the weights for the FC layers, bias is set to 0. 
        As per ICLR paper.
        """
        for m in self.layers:
            # If the layer is a conv layer, initialise the weights and set the bias to 0
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu')) 
                if m.bias is not None: 
                    init.constant_(m.bias, 0) 
            # If the layer is a linear layer, initialise the weights and set the bias to 0
            elif isinstance(m, nn.Linear): 
                gain = 0.01 if m.out_features == 1 else nn.init.calculate_gain('relu')
                init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    
    def print_weights(self):
        """
        Print the weights of the Actor.
        """
        for name, param in self.layers.named_parameters():
            print(f"Actor layer: {name}")
            print(param.data)  # or param for the raw tensor

    def forward(self, x):
        """
        Forward pass of the Actor.
        """
        return self.layers(x)

    def save_model(self, path="training/models/actor_model.pth"):
        """
        Save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, pathname="training/models/actor_model.pth"):
        """
        Load the model.
        """
        if pathname is None or pathname == "None":
            return
        state_dict = torch.load(pathname, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)

# --------- Critic Network ---------
class Critic(nn.Module):
    def __init__(self):
        """
        Initialise the Critic.
        """
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, padding=1), # 4 channels for the input image, 32 filters, kernel size 3, padding 1
        nn.ReLU(), # ReLU activation function
        nn.MaxPool2d(2), # Max pooling with a pool size of 2

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.AdaptiveAvgPool2d((5, 5)),  # Output shape fixed to (128, 5, 5)
        
        nn.Flatten(1),
        nn.Linear(128 * 5 * 5, 512), 
        nn.ReLU(),
        nn.Linear(512, 1)) 
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5) # LR as per ICLR paper
        
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialisation of the weights for the FC layers, bias is set to 0. 
        As per ICLR paper.
        """
        for m in self.layers:
            # If the layer is a conv layer, initialise the weights and set the bias to 0
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # If the layer is a linear layer, initialise the weights and set the bias to 0
            elif isinstance(m, nn.Linear):
                gain = 0.01 if m.out_features == 1 else nn.init.calculate_gain('relu')
                init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def print_weights(self):    
        """
        Print the weights of the Critic.
        """
        for name, param in self.layers.named_parameters():
            print(f"Critic layer: {name}")
            print(param.data)  # or param for the raw tensor


    def forward(self, x):
        """
        Forward pass of the Critic.
        """
        return self.layers(x)

    def save_model(self, path="training/models/critic_model.pth"):
        """
        Save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_model(self, pathname="training/models/critic_model.pth"):
        """
        Load the model.
        """
        if pathname is None or pathname == "None":
            return
        state_dict = torch.load(pathname, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)