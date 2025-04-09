"""
This file will contain the classes for the neural nets  
    
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten

class NeuralNets:
    
    def __init__ (self):
        
        self.model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
        ])
        
    def compile (self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics = ['accuracy'])
        
    def train (self,input):
        epochs = 120
        history = self.model.fit(
            input,
            epochs=epochs
        )
        return history



# Ignore the classes below for now


class Actor (NeuralNets):
    
    def __init__ (self, base_model_output, outputs):
        super().__init__(base_model_output)
        self.output = self.final_layer(self.base_model_output)
        
    def final_layer (self, action_space, base_model_output):
        return layers.Dense(action_space.n, activation='softmax')(base_model_output)

class Critic (NeuralNets):
    
    def __init__ (self, base_model_output, outputs):
        super().__init__(base_model_output)
        self.output = self.final_layer(self.base_model_output)
        
    def final_layer (self, base_model_output):
        return layers.Dense(1, activation='softmax')(base_model_output)
    