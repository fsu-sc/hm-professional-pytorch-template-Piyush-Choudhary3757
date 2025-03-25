import torch
import numpy as np
from torch.utils.data import Dataset
from base import BaseDataLoader

class FunctionDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function
        self.x = torch.FloatTensor(n_samples, 1).uniform_(0, 2 * np.pi)
        self.generate_data()
        self.normalize_data()

    def generate_data(self):
        # Generate random noise
        noise = torch.FloatTensor(self.n_samples, 1).uniform_(-1, 1)
        
        if self.function == 'linear':
            self.y = 1.5 * self.x + 0.3 + noise
        elif self.function == 'quadratic':
            self.y = 2 * self.x**2 + 0.5 * self.x + 0.3 + noise
        elif self.function == 'harmonic':
            self.y = 0.5 * self.x**2 + 5 * torch.sin(self.x) + 3 * torch.cos(3 * self.x) + 2 + noise
        else:
            raise ValueError(f"Unknown function type: {self.function}")

    def normalize_data(self):
        # Normalize x
        self.x_mean = torch.mean(self.x)
        self.x_std = torch.std(self.x)
        self.x = (self.x - self.x_mean) / self.x_std

        # Normalize y
        self.y_mean = torch.mean(self.y)
        self.y_std = torch.std(self.y)
        self.y = (self.y - self.y_mean) / self.y_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        self.dataset = FunctionDataset(n_samples, function)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)