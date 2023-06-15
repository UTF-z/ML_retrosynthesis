import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pickle as pkl
from tqdm import tqdm
from utils.utils import count_parameters, save_best, load_best
from model.networks import MLP, PretrainedRegresser
import argparse
import numpy as np
import torch.nn.functional as F

def build_data_task2(root_dir=os.path.join(".","MoleculeEvaluationData"),unpack=True):
    train_path = os.path.join(root_dir,"train.pkl")
    train_pkl = open(train_path,"rb")
    test_path = os.path.join(root_dir,"test.pkl")
    test_pkl = open(test_path,"rb")
    train_data = pkl.load(train_pkl)
    test_data = pkl.load(test_pkl)
    train_pkl.close()
    test_pkl.close()
    train_size = train_data['packed_fp'].shape[0]
    test_size = test_data['packed_fp'].shape[0]
    if unpack:
        train_data = (torch.tensor(np.unpackbits(train_data['packed_fp']).reshape(train_size,-1)),train_data['values']) # (uint8, fp32)
        test_data = (torch.tensor(np.unpackbits(test_data['packed_fp']).reshape(test_size,-1)),test_data['values']) # (uint8, fp32)
    else:
        train_data = (torch.tensor(train_data['packed_fp']),train_data['values']) # (uint8, fp32)
        test_data = (torch.tensor(test_data['packed_fp']),test_data['values']) # (uint8, fp32)
    return train_data, test_data

class Task2(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        data, label = data
        data = data.to(torch.float32)
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.data.shape[0]
    
def build_loader_task2(train_data,test_data,val_ratio=0.1,batch_size=512):
    trainset = Task2(train_data)
    val_size = int(len(trainset)*val_ratio)
    train_size = len(trainset) - val_size
    if val_size != 0.:
        trainset, valset = random_split(trainset,
                    [train_size,val_size],
                    torch.Generator().manual_seed(seed))
    else:
        valset = None
    testset = Task2(test_data)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True) if not (valset is None) else None
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)
    return train_loader, val_loader, test_loader

@torch.no_grad()
def test(model, loader, device="cpu"):
    model.eval()
    cnt = []
    avg_loss = []
    for data,label in loader:
        data=data.to(device=device)
        label=label.to(device=device)
        pred = model(data)
        loss = F.l1_loss(pred,label)
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
    parser.add_argument("--batch",type=int,default=2048)
    parser.add_argument("--episode",type=int,default=1000)
    parser.add_argument("--lr",type=float,default=5e-4)
    parser.add_argument("--w",type=float,default=5e-4)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--pretrain",action="store_true",default=False)
    args=parser.parse_args()
    val_ratio = 0.
    batch_size = args.batch
    episode = args.episode
    lr = args.lr
    weight_decay = args.w
    device = args.device
    pretrain = args.pretrain
    # build data
    train_data, test_data = build_data_task2(os.path.join(".","resources","MoleculeEvaluationData"))
    in_dim = train_data[0].shape[1]
    train_loader, val_loader, test_loader = build_loader_task2(train_data,
                                                               test_data,
                                                               val_ratio,
                                                               batch_size)
    # logger
    log_dir = os.path.join(".","log","task2","pretrain" if pretrain else "baseline")
    ckpt_dir = os.path.join(".","checkpoint","task2","pretrain" if pretrain else "baseline")
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(ckpt_dir,exist_ok=True)
    train_logger = open(os.path.join(log_dir,"train_logger.txt"),"wt")
    val_logger = open(os.path.join(log_dir,"val_logger.txt"),"wt")
    # build model
    if not pretrain:
        model = MLP(sizes=[in_dim,512,128,1],dropout=0.1,act_func="elu",last_act="elu")
        # model = PretrainedRegresser(in_dim,act_func="relu")
    else:
        model = PretrainedRegresser(in_dim,act_func="relu")
        pretrain_dir = os.path.join(".","checkpoint","pretrain")
        model.embed.load_state_dict(load_best(os.path.join(pretrain_dir,"baseline_best.pth")))
    model.to(device=device)
    mse = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    best_loss = 100.
    val_loss = 0.
    tbar = tqdm(range(episode))
    for e in range(episode):
        train_cnt = []
        train_loss = []
        for data, label in train_loader:
            data=data.to(device=device)
            label=label.to(device=device)
            optimizer.zero_grad()
            pred = model(data)
            loss = mse(pred,label)
            loss.backward()
            optimizer.step()
            train_cnt.append(data.shape[0])
            train_loss.append(loss.item())
        avg_loss = sum([train_loss[j] * train_cnt[j] for j in range(len(train_cnt))]) / sum(train_cnt)
        print(e,' ',avg_loss, file=train_logger)
        if (e+1) % 1 == 0 or e == 0:
            with torch.no_grad():
                val_loss,_ = test(model,test_loader,device=device)
            if val_loss <= best_loss:
                best_loss = val_loss
                save_best(model,os.path.join(ckpt_dir,"baseline_best.pth"))
            print(e, ' ', val_loss, file=val_logger)
        tbar.set_postfix_str("train loss: {0:.5f}, val loss: {1:.5f}".format(avg_loss,val_loss))  
        tbar.update()

    with torch.no_grad():
        model.load_state_dict(load_best(os.path.join(ckpt_dir,"baseline_best.pth")))
        test_loss,_ = test(model,test_loader,device=device)
    print("Finally, Test Loss:", ' ', test_loss)
    print("Finally, Test Loss:", ' ', test_loss, file=train_logger)
    print("Finally, Test Loss:", ' ', test_loss, file=val_logger)
    train_logger.close()
    val_logger.close()