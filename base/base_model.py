import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, *input):
        """
        Forward pass logic
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)