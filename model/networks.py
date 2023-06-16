import os
import sys
sys.path.append(os.path.abspath("."))
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.utils import is_compatible
from tqdm import tqdm
import torch.multiprocessing as mp
from utils.utils import get_fp

act_funcs={
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax
}

class MLP(nn.Module):

    def __init__(self, sizes, dropout=0., act_func="relu", last_act=None) -> None:
        super().__init__()
        self.net = nn.Sequential()
        self.device = 'cpu'
        for i, (in_size, out_size) in enumerate(zip(sizes[0:-1], sizes[1:])):
            self.net.add_module(f'fc_{i}', nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                self.net.add_module(f'dropout_{i}', nn.Dropout(p=dropout))
                self.net.add_module(f'{act_func}_{i}', act_funcs[act_func]())
        if last_act is not None:
            self.net.add_module(f'{last_act}_output', act_funcs[last_act]())
    
    def to(self, device):
        super().to(device=device)
        self.device = device
        return self

    def make_task2_infer_fn(self):

        def inference(smile: str):
            fp, _ = get_fp(smile)
            fp = np.expand_dims(fp, axis=0)
            fp = torch.FloatTensor(fp).to(self.device)
            value = self.forward(fp).cpu()
            value = value.squeeze(0)
            return value
        return inference

    
    def forward(self, x):
        return self.net(x)

class AutoEncoder(nn.Module):
    def __init__(self, dim, code_dim, dropout=0) -> None:
        super().__init__()
        self.encoder = MLP(sizes=[dim,512,code_dim],act_func="relu", dropout=dropout)
        self.decoder = MLP(sizes=[code_dim,512,dim],act_func="relu", dropout=dropout)
        self.ori_forward = self.forward

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    def encode(self,x):
        return self.encoder(x)
    
    def embed(self):
        self.forward=self.encode

    def retrain(self):
        self.forward=self.ori_forward

class Baseline(nn.Module):

    @classmethod
    def load_and_construct(cls, path):
        with open(path, 'rb') as f:
            ckpt = torch.load(f) 
            config = ckpt['config']
            state_dict = ckpt['state_dict']
            model = cls(config)
            model.load_state_dict(state_dict)
        return model

    def __init__(self, config) -> None:
        super().__init__()
        sizes = config['sizes']
        dropout = config['dropout']
        weight_decay = config['weight_decay']
        self.device = config['device']
        self.topk = config['topk']
        self.filter = config['filter']
        self.hash_table = config['hash_table']
        self.num_worker = config['num_worker']
        self.lr = config['lr']
        self.multi_processing = config.get('multi_processing', True)
        self.config = config
        self.net = MLP(sizes, dropout=dropout).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.lr, weight_decay=weight_decay)
    
    def forward(self, pack, mode='train'):
        if mode == 'train':
            return self.train_step(pack)
        elif mode == 'val':
            return self.val_step(pack)
        elif mode == 'infer':
            return self.inference(pack)

    def train_step(self, pack):
        self.net.train()
        fp = pack['fingerprint'].to(self.device)
        gt = pack['template_idx'].to(self.device)
        out = self.net(fp)
        loss = self.compute_loss(out, gt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        _, label = torch.topk(out, 
                           k = 1, 
                           dim=1, 
                           largest=True)
        train_correct_cnt = (label == gt.reshape(-1,1).repeat(1,1)).flatten().sum()
        train_total_cnt = len(label)
        return loss.item(), train_correct_cnt, train_total_cnt, {}
    
    def compute_loss(self, pred, gt):
        return F.cross_entropy(pred, gt)
    
    @torch.no_grad()
    def val_step(self, pack):
        self.net.eval()
        fp = pack['fingerprint'].to(self.device)
        out = self.net(fp)
        filterK = self.topk * 5 if self.filter else self.topk
        _, label = torch.topk(out, 
                    k = filterK, 
                    dim=1, 
                    largest=True)
        if not self.filter:
            gt = pack['template_idx'].to(self.device)
            val_correct_cnt = (label == gt.reshape(-1,1).repeat(1,self.topk)).flatten().sum()
        else:
            B = label.shape[0]
            gt = pack['template_idx']
            label = label.cpu()
            prelabel = label[:,:self.topk]
            if not self.multi_processing:
                start = 0
                end = len(label)
                self._is_compatible(label, prelabel, pack['smile'], self.hash_table, start=start, end=end)
            else:
                label = label.share_memory_()
                prelabel = prelabel.share_memory_()
                step = (label.shape[0] + self.num_worker - 1) // self.num_worker
                tasks = [mp.Process(target=self._is_compatible, args=(label, prelabel, pack['smile'], self.hash_table, j*step, min(label.shape[0], (j+1)*step))) for j in range(min(B, self.num_worker))]
                [p.start() for p in tasks]
                [p.join() for p in tasks]
            label = prelabel
            val_correct_cnt = (label == gt.reshape(-1,1).repeat(1,self.topk)).flatten().sum()
        val_total_cnt = len(label)
        return val_correct_cnt, val_total_cnt
    
    @torch.no_grad()
    def inference(self, pack):
        self.net.eval()
        fp = pack['fingerprint'].to(self.device)
        out = self.net(fp)
        filterK = self.topk * 5 if self.filter else self.topk
        out, label = torch.topk(out, 
                    k = filterK, 
                    dim=1, 
                    largest=True)
        if not self.filter:
            out = out[:, :self.topk]
            out = F.softmax(out, dim=1)
            label = label[:, :self.topk]
            return out.cpu(), label.cpu()
        else:
            B = label.shape[0]
            out = out[:, :self.topk]
            out = F.softmax(out, dim=1)
            label = label.cpu()
            prelabel = label[:, :self.topk]
            if not self.multi_processing:
                start = 0
                end = len(label)
                self._is_compatible(label, prelabel, pack['smile'], self.hash_table, start=start, end=end)
            else:
                label = label.share_memory_()
                prelabel = prelabel.share_memory_()
                step = (label.shape[0] + self.num_worker - 1) // self.num_worker
                tasks = [mp.Process(target=self._is_compatible, args=(label, prelabel, pack['smile'], self.hash_table, j*step, min(label.shape[0], (j+1)*step))) for j in range(min(B, self.num_worker))]
                [p.start() for p in tasks]
                [p.join() for p in tasks]
            label = prelabel
        return out.cpu(), label

    def _is_compatible(self, label, prelabel, smile, hash_table, start, end):
        for i in range(start, end):
            itr = 0
            for k in range(label.shape[1]):
                if itr >= self.topk:
                    break
                if is_compatible(smile[i], hash_table[label[i, k]]):
                    prelabel[i, itr] = label[i, k]
                    itr += 1
        return 0
    
    def save(self, path):
        dump_file = {
            'config': self.config,
            'state_dict': self.state_dict()
        }
        with open(path, 'wb') as f:
            torch.save(dump_file, f)
    
    def load_ckpt(self, path):
        with open(path, 'rb') as f:
            ckpt = torch.load(f)
        state_dict = ckpt['state_dict']
        self.load_state_dict(state_dict)

    def make_task1_infer_fn(self):

        def inference(smile: str):
            fp, _ = get_fp(smile)
            fp = np.expand_dims(fp, axis=0)
            fp = torch.FloatTensor(fp)
            pack = {
                'fingerprint': fp,
                'smile': smile
            }
            prob, label = self.inference(pack)
            prob = prob.numpy().squeeze(axis=0)
            costs = -np.log(prob)
            label = label.numpy().squeeze(axis=0)
            temps = [self.hash_table[i] for i in label]
            return temps, costs
        return inference

class PretrainedRegresser(nn.Module):
    def __init__(self,in_dim,act_func="relu") -> None:
        super().__init__()
        self.embed = AutoEncoder(in_dim,50)
        self.embed.embed()
        self.decoder = MLP(sizes=[50,128,1],act_func="relu")

    def forward(self,x):
        return self.decoder(self.embed(x))

class AEClassifier(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        input_dim = config['input_dim']
        latent_dim = config['latent_dim']
        header_size = config['header_size']
        header_dropout = config['header_dropout']
        lr = config['lr']
        self.device = config['device']
        self.ae = AutoEncoder(input_dim, latent_dim)
        self.header = MLP(header_size, dropout=header_dropout)
        self.header_optimizer = Adam(self.header.parameters(), lr=lr)
        self.ae_optimizer = Adam(self.ae.parameters(), lr=lr)
    
    def forward(self, pack, mode='train'):
        if mode == 'train':
            return self.train_step(pack)
        elif mode == 'val':
            return self.val_step(pack)
    
    def train_step(self, pack):
        fingerprint = pack['fingerprint'].to(self.device)
        gt = pack['template_idx'].to(self.device)
        recon, z = self.ae(fingerprint)
        pred = self.header(z)
        # recon loss
        L_rec = torch.norm(fingerprint-recon, p=1, dim=-1).mean()
        # cls loss
        L_cls = F.cross_entropy(pred, gt)
        # reg loss
        L_reg = (z**2).sum()
        # dist loss
        gt1 = torch.unsqueeze(gt, dim=0)
        gt2 = torch.unsqueeze(gt, dim=1)
        mask = (gt1 - gt2) == 0
        z1 = torch.unsqueeze(z, dim=0)
        z2 = torch.unsqueeze(z, dim=1)
        z_dist = torch.norm(z1-z2, p=2, dim=-1, keepdim=False)
        L_dist = torch.max(z_dist * mask + (1-z_dist) * ~mask).sum()
        loss = L_rec + L_cls + L_dist + L_reg
        self.header_optimizer.zero_grad()
        self.ae_optimizer.zero_grad()
        loss.backward()
        self.header_optimizer.step()
        self.ae_optimizer.step()
        # train log
        pred = torch.argmax(pred, dim=-1)
        correct_num = (pred == gt).sum()
        total_num = len(pred)
        loss_info = {
            'L_rec': L_rec.item(),
            'L_cls': L_cls.item(),
            'L_dist': L_dist.item(),
            'L_reg': L_reg.item()
        }
        return loss.item(), correct_num, total_num, loss_info
    
    def val_step(self, pack):
        fingerprint = pack['fingerprint'].to(self.device)
        gt = pack['template_idx'].to(self.device)
        recon, z = self.ae(fingerprint)
        pred = self.header(z)
        pred = torch.argmax(pred, dim=-1)
        correct_num = (pred == gt).sum()
        total_num = len(pred)
        return correct_num, total_num
    
    def save(self, path):
        with open(path, 'wb') as f:
            torch.save(self.state_dict(), f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)



