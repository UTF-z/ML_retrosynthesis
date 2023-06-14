import pickle
import numpy 
import torch
from torch.utils.data.dataset import Dataset

class SingleStep(Dataset):

    def __init__(self, data_path, temp_hash_path) -> None:
        super().__init__()
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        with open(temp_hash_path, 'rb') as f:
            self.temp_hash = pickle.load(f)
        self.template_num = len(self.temp_hash)
        self.ids = data['ids']
        self.classes = data['classes']
        self.fingerprints = data['fingerprints']
        self.templates = data['templates']
    
    def __len__(self):
        return len(self.fingerprints)
    
    def __getitem__(self, idx):
        pack = {
            'id': self.ids[idx],
            'class': self.classes[idx],
            'fingerprint': torch.FloatTensor(self.fingerprints[idx]),
            'template': self.templates[idx],
            'template_idx': self.temp_hash[self.templates[idx]]
        }
        return pack



