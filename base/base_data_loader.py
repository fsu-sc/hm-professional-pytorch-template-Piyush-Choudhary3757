from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        
        self.n_samples = len(dataset)
        
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        
        super().__init__(sampler=self.sampler, **self.init_kwargs)
        
    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
            
        idx_full = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(idx_full)
            
        len_valid = int(self.n_samples * split)
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        
        return train_sampler, valid_sampler
        
    @property
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)