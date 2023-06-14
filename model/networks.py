from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class MLP(nn.Module):

    def __init__(self, sizes, dropout=0) -> None:
        super().__init__()
        self.net = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(sizes[0:-1], sizes[1:])):
            self.net.add_module(f'fc_{i}', nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                self.net.add_module(f'dropout_{i}', nn.Dropout(p=dropout))
                self.net.add_module(f'relu_{i}', nn.ReLU())
    
    def forward(self, x):
        return self.net(x)

class Baseline(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        sizes = config['sizes']
        dropout = config['dropout']
        weight_decay = config['weight_decay']
        self.device = config['device']
        self.net = MLP(sizes, dropout=dropout)
        self.optimizer = Adam(self.net.parameters(), lr=1e-3, weight_decay=weight_decay)
    
    def forward(self, pack, mode='train'):
        if mode == 'train':
            return self.train_step(pack)
        elif mode == 'val':
            return self.val_step(pack)

    def train_step(self, pack):
        self.net.train()
        fp = pack['fingerprint'].to(self.device)
        gt = pack['template_idx'].to(self.device)
        out = self.net(fp)
        loss = F.cross_entropy(out, gt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        label = torch.argmax(out, dim=1, keepdim=False)
        train_correct_cnt = (label == gt).sum()
        train_total_cnt = len(label)
        return loss.item(), train_correct_cnt, train_total_cnt
    
    @torch.no_grad()
    def val_step(self, pack):
        self.net.eval()
        fp = pack['fingerprint'].to(self.device)
        gt = pack['template_idx'].to(self.device)
        out = self.net(fp)
        label = torch.argmax(out, dim=1, keepdim=False)
        val_correct_cnt = (label == gt).sum()
        val_total_cnt = len(label)
        return val_correct_cnt, val_total_cnt
    
    def save(self, path):
        with open(path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)