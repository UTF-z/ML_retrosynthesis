import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pickle as pkl
from tqdm import tqdm
from utils.utils import count_parameters, save_best, load_best
from model.networks import MLP, AutoEncoder
import argparse
import numpy as np
import torch.nn.functional as F

def build_data_pretrain(root_dir=os.path.join(".","MoleculeEvaluationData")):
    path = os.path.join(root_dir,"AE_data.pkl")
    train_pkl = open(path,"rb")
    train_data = pkl.load(train_pkl)
    return torch.tensor(train_data['fingerprints'])

class Task2(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        data = data.to(torch.float32)
        self.data = data

    def __getitem__(self, index):
        return self.data[index], self.data[index]
    
    def __len__(self):
        return self.data.shape[0]
    
def build_loader_pretrain(train_data,val_ratio=0.1,batch_size=512):
    trainset = Task2(train_data)
    val_size = int(len(trainset)*val_ratio)
    train_size = len(trainset) - val_size
    if val_size != 0:
        trainset, valset = random_split(trainset,
                    [train_size,val_size],
                    torch.Generator().manual_seed(seed))
    else:
        valset = None
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True) if not (valset is None) else None
    return train_loader, val_loader, val_loader

@torch.no_grad()
def test(model, loader, device="cpu"):
    model.eval()
    cnt = []
    avg_loss = []
    for data,_ in loader:
        data=data.to(device=device)
        pred, _ = model(data)
        loss = F.l1_loss(pred,data)
        cnt.append(data.shape[0])
        avg_loss.append(loss.item())
    model.train()
    return sum([avg_loss[j] * cnt[j] for j in range(len(cnt))]) / sum(cnt), None

if __name__ == "__main__":
    seed=8192
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # parser
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",type=int,default=1024)
    parser.add_argument("--episode",type=int,default=50)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--w",type=float,default=0.)
    parser.add_argument("--device",type=str,default="cuda")
    args=parser.parse_args()
    val_ratio = 0.1
    batch_size = args.batch
    episode = args.episode
    lr = args.lr
    weight_decay = args.w
    device = args.device
    # build data
    train_data = build_data_pretrain(os.path.join(".","resources","pretrain"))
    in_dim = 2048
    train_loader, val_loader, test_loader = build_loader_pretrain(train_data,
                                                               val_ratio,
                                                               batch_size)
    # logger
    log_dir = os.path.join(".","log","pretrain")
    ckpt_dir = os.path.join(".","checkpoint","pretrain")
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(ckpt_dir,exist_ok=True)
    train_logger = open(os.path.join(log_dir,"train_logger.txt"),"wt")
    val_logger = open(os.path.join(log_dir,"val_logger.txt"),"wt")
    # build model
    model = AutoEncoder(in_dim,50)
    model.to(device=device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    best_loss = 100.
    val_loss = 0.
    tbar = tqdm(range(episode))
    for e in range(episode):
        train_cnt = []
        train_loss = []
        for data, _ in train_loader:
            data=data.to(device=device)
            optimizer.zero_grad()
            pred, z = model(data)
            loss = criterion(pred,data)
            loss.backward()
            optimizer.step()
            train_cnt.append(data.shape[0])
            train_loss.append(loss.item())
        avg_loss = sum([train_loss[j] * train_cnt[j] for j in range(len(train_cnt))]) / sum(train_cnt)
        print(e,' ',avg_loss, file=train_logger)
        if (e+1) % 5 == 0 or e == 0:
            with torch.no_grad():
                val_loss,_ = test(model,val_loader,device=device)
            if val_loss <= best_loss:
                best_loss = val_loss
                save_best(model,os.path.join(ckpt_dir,"baseline_best.pth"))
            print(e, ' ', val_loss, file=val_logger)
        tbar.set_postfix_str("train loss: {0:.5f}, val loss: {1:.5f}".format(avg_loss,val_loss))  
        tbar.update()

    with torch.no_grad():
        test_loss,_ = test(model,test_loader,device=device)
    print("Finally, Test Loss:", ' ', test_loss, file=train_logger)
    print("Finally, Test Loss:", ' ', test_loss, file=val_logger)
    train_logger.close()
    val_logger.close()