import sys
sys.path.append('.')
from pathlib import Path
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

    def __init__(self, sizes) -> None:
        super().__init__()
        self.net = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(sizes[0:-1], sizes[1:])):
            self.net.add_module(f'fc_{i}', nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                self.net.add_module(f'relu_{i}', nn.ReLU())
    
    def forward(self, x):
        return self.net(x)

EPOCH = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.add('log/task1_baseline_train.log', filter=lambda record: record['extra']['name'] == 'train')
logger.add('log/task1_baseline_test.log', filter=lambda record: record['extra']['name'] == 'test')
log_train = logger.bind(name='train')
log_test = logger.bind(name='test')
data_dir = Path('sources')
train_set = SingleStep(data_path=data_dir/'task1_train.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
val_set = SingleStep(data_path=data_dir/'task1_val.pkl', temp_hash_path=data_dir/'temp_hash.pkl')
train_loader = DataLoader(train_set, batch_size=512, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False, drop_last=False)
model = MLP([2048, 512, train_set.template_num]).to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-3)
with tqdm(range(EPOCH)) as tbar:
    for i in range(EPOCH):
        loss_lst = []
        correct_cnt = 0
        total_cnt = 0
        for pack in train_loader:
            fp = pack['fingerprint'].to(DEVICE)
            gt = pack['template_idx'].to(DEVICE)
            out = model(fp)
            loss = F.cross_entropy(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train log
            loss_lst.append(loss.item())
            label = torch.argmax(out, dim=1, keepdim=False)
            correct_cnt += (label == gt).sum()
            total_cnt += len(label)
        avg_loss = np.mean(loss_lst)
        avg_acc = correct_cnt / total_cnt
        tbar.set_postfix_str(f'loss: {np.mean(loss_lst)}, acc: {avg_acc}')
        log_train.info(f'loss: {np.mean(loss_lst)}, acc: {avg_acc}')
        tbar.update()
log_train.critical('train finished') 

correct_cnt = 0
total_cnt = 0
with tqdm(val_loader) as vbar:
    for batch in val_loader:
        fp = pack['fingerprint'].to(DEVICE)
        gt = pack['template_idx'].to(DEVICE)
        out = model(fp)
        # val log
        label = torch.argmax(out, dim=1, keepdim=False)
        correct_cnt += (label == gt).sum()
        total_cnt += len(label)
        vbar.update()
log_test.critical(f'val acc: {correct_cnt / total_cnt}')

# save checkpoint
now = time.strftime("%d-%H%M%S", time.localtime(time.time()))
name = now + "_mlp.pkl"
with open(Path('checkpoint')/name, 'wb') as f:
    torch.save(model.state_dict(), f)
    





