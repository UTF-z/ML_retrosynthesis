import pickle
import numpy 
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path

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
    
    def _vis(self):
        fp_distribution = {}
        total_cnt = 0
        for i in range(len(self.fingerprints)):
            fp = self.fingerprints[i]
            temp = self.templates[i]
            temp_id = self.temp_hash[temp]
            fp_distribution[temp_id] = fp_distribution.get(temp_id, 0) + 1
            total_cnt += 1
        print(total_cnt)
        return fp_distribution
        
def vis_distribution():
    import matplotlib.pyplot as plt
    data = SingleStep(Path('resources')/'task1_train.pkl', temp_hash_path=Path('resources')/'temp_hash.pkl')
    fp_distribution = data._vis()
    plt.bar(fp_distribution.keys(), fp_distribution.values())
    plt.savefig('./pdf.png')

if __name__ == '__main__':
    vis_distribution()





