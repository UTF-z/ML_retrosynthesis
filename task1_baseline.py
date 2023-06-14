import sys
sys.path.append('.')
from pathlib import Path
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from dataset.single_step import SingleStep
from tqdm import tqdm
from loguru import logger

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
        self.net = MLP(sizes, dropout=dropout)
        self.optimizer = Adam(self.net.parameters(), lr=1e-3, weight_decay=weight_decay)
    
    def forward(self, pack, mode='train'):
        if mode == 'train':
            return self.train_step(pack)
        elif mode == 'val':
            return self.val_step(pack)

    def train_step(self, pack):
        self.net.train()
        fp = pack['fingerprint'].to(DEVICE)
        gt = pack['template_idx'].to(DEVICE)
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
        fp = pack['fingerprint'].to(DEVICE)
        gt = pack['template_idx'].to(DEVICE)
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

EPOCH = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_INT = 10
DROPOUT = 0.5
now = time.strftime("%d-%H%M%S", time.localtime(time.time()))
# log setting
train_log_path = f'log/{now}_task1_baseline_train.log'
val_log_path = f'log/{now}_task1_baseline_val.log'
test_log_path = f'log/{now}_task1_baseline_test.log'
try:
    os.remove(train_log_path)
except Exception as e:
    pass
try:
    os.remove(test_log_path)
except Exception as e:
    pass
try:
    os.remove(val_log_path)
except Exception as e:
    pass
logger.remove(None)
logger.add(train_log_path, filter=lambda record: record['extra']['name'] == 'train')
logger.add(val_log_path, filter=lambda record: record['extra']['name'] == 'val')
logger.add(test_log_path, filter=lambda record: record['extra']['name'] == 'test')
log_train = logger.bind(name='train')
log_test = logger.bind(name='test')
log_val = logger.bind(name='val')

# get data
data_dir = Path('resources')
train_set = SingleStep(data_path=data_dir/'task1_train.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
val_set = SingleStep(data_path=data_dir/'task1_val.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
test_set = SingleStep(data_path=data_dir/'task1_test.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
train_loader = DataLoader(train_set, batch_size=512, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, drop_last=False)

# get model
model_name = now + "_mlp.pkl"
state_dict_path = Path('checkpoint') / model_name
model_config = {
    'sizes': [2048, 512, 512, train_set.template_num],
    'dropout': DROPOUT,
    'weight_decay': 1e-4
}
model = Baseline(model_config).to(DEVICE)

# train
best_val_acc = 0
with tqdm(range(EPOCH)) as train_bar:
    for i in range(EPOCH):
        loss_lst = []
        train_correct_cnt = 0
        train_total_cnt = 0
        for pack in train_loader:
            loss, correct_num, total_num = model(pack, mode='train')
            # train log
            loss_lst.append(loss)
            train_correct_cnt += correct_num
            train_total_cnt += total_num
        avg_loss = np.mean(loss_lst)
        avg_acc = train_correct_cnt / train_total_cnt
        train_bar.set_postfix_str(f'loss: {np.mean(loss_lst)}, acc: {avg_acc}')
        log_train.info(f'loss: {np.mean(loss_lst)}, acc: {avg_acc}')
        train_bar.update()
        # validation
        if i % VAL_INT == 0:
            val_correct_cnt = 0
            val_total_cnt = 0
            for pack in val_loader:
                correct_num, total_num = model(pack, mode='val')
                val_correct_cnt += correct_num
                val_total_cnt += total_num
            val_acc = val_correct_cnt / val_total_cnt
            if val_acc > best_val_acc:
                model.save(state_dict_path)
            log_val.critical(f'epoch: {i}, val_acc: {val_acc}')

log_train.critical('train finished') 
model.load(state_dict_path)
test_correct_cnt = 0
test_total_cnt = 0
with tqdm(test_loader) as testbar:
    for pack in test_loader:
        correct_num, total_num = model(pack, mode='val')
        test_correct_cnt += correct_num
        test_total_cnt += total_num
        testbar.update()
log_test.critical(f'test acc: {test_correct_cnt / test_total_cnt}')
    





