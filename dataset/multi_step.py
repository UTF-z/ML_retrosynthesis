import sys
sys.path.append('.')
import os
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data.dataset import Dataset

class MultiStep(Dataset):

    def __init__(self, data_root) -> None:
        super().__init__()
        data_dir = Path(data_root)
        with open(data_dir / 'starting_mols.pkl', 'rb') as f:
            self.starting_mols = pickle.load(f)
        with open(data_dir / 'test_mols.pkl', 'rb') as f:
            self.test_mols = pickle.load(f)
        with open(data_dir / 'target_mol_route.pkl', 'rb') as f:
            self.target_mol_route = pickle.load(f)
    
    def __len__(self):
        return len(self.test_mols)
    
    def __getitem__(self, idx):
        return self.test_mols[idx], self.target_mol_route[idx]
    
    def make_test_fn(self):
        def test_in(smile):
            return smile in self.starting_mols
        return test_in

if __name__ == '__main__':
    data = MultiStep('/root/final/resources/Multi-Step task')
    print(data[0])
