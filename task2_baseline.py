import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pickle as pkl
from tqdm import tqdm
from utils import count_parameters, save_best, load_best
from model import MLP
import argparse
import numpy as np

def build_data_task2(root_dir=os.path.join(".","MoleculeEvaluationData"),unpack=True):
    train_path = os.path.join(root_dir,"train.pkl")
    train_pkl = open(train_path,"rb")
    test_path = os.path.join(root_dir,"test.pkl")
    test_pkl = open(test_path,"rb")
    train_data = pkl.load(train_pkl)
    test_data = pkl.load(test_pkl)
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
    if val_size != 0:
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
        loss = mse(pred,label)
        cnt.append(data.shape[0])
        avg_loss.append(loss.item())
    model.train()
    return sum([train_loss[j] * train_cnt[j] for j in range(len(train_cnt))]) / sum(train_cnt), None

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
    parser.add_argument("--device",type=str,default="cpu")
    args=parser.parse_args()
    val_ratio = 0.1
    batch_size = args.batch
    episode = args.episode
    lr = args.lr
    weight_decay = args.w
    device = args.device
    # build data
    train_data, test_data = build_data_task2(os.path.join(".","MoleculeEvaluationData"))
    in_dim = train_data[0].shape[1]
    train_loader, val_loader, test_loader = build_loader_task2(train_data,
                                                               test_data,
                                                               val_ratio,
                                                               batch_size)
    # logger
    train_logger = open("train_logger.txt","wt")
    val_logger = open("val_logger.txt","wt")
    # build model
    model = MLP(in_dim=in_dim,hid_dim=[1024,512],out_dim=1,act_func="relu")
    model.to(device=device)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    tbar = tqdm(range(episode))
    for e in range(episode):
        best_loss = 100.
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
        print(e,' ',sum([train_loss[j] * train_cnt[j] for j in range(len(train_cnt))]) / sum(train_cnt), file=train_logger)
        if e % 5 == 0:
            with torch.no_grad():
                val_loss,_ = test(model,val_loader,device=device)
            if val_loss <= best_loss:
                save_best(model,"baseline_best.pth")
            print(e, ' ', val_loss, file=val_logger)
        tbar.set_postfix_str("loss: {0:.5f}".format(best_loss))  
        tbar.update()

    with torch.no_grad():
        test_loss,_ = test(model,test_loader,device=device)
    print("Finally, Test Loss:", ' ', test_loss, file=train_logger)
    print("Finally, Test Loss:", ' ', test_loss, file=val_logger)