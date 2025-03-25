import torch
import torch.nn as nn
from base import BaseModel

class DynamicModel(BaseModel):
    def __init__(self, num_hidden_layers=2, hidden_sizes=[64, 32], 
                 hidden_activation='relu', output_activation='linear'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer (1 feature to first hidden layer)
        self.layers.append(nn.Linear(1, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        # Set activation functions
        self.hidden_activation = self._get_activation(hidden_activation)
        self.output_activation = self._get_activation(output_activation)
        
    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x):
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.view(-1, 1)
        
        # Forward pass through hidden layers
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
            
        # Output layer
        x = self.output_activation(self.layers[-1](x))
        return x