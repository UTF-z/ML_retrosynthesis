import torch
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils.utils import dataspliter


def rebuild_task2(root_dir=os.path.join(".","MoleculeEvaluationData")):
    train_path = os.path.join(root_dir,"train.pkl")
    train_pkl = open(train_path,"rb")
    test_path = os.path.join(root_dir,"test.pkl")
    test_pkl = open(test_path,"rb")
    train_data = pkl.load(train_pkl)
    test_data = pkl.load(test_pkl)
    train_pkl.close()
    test_pkl.close()
    train_data = (train_data['packed_fp'],train_data['values']) # (uint8, fp32)
    test_data = (test_data['packed_fp'],test_data['values']) # (uint8, fp32)
    data_all = torch.cat([torch.tensor(train_data[0]),
                         torch.tensor(test_data[0])],
                         dim=0)
    label_all = torch.cat([train_data[1],
                          test_data[1]],
                          dim=0)
    train_data, test_data = dataspliter(data_all,label_all,0.8)
    save_new_dataset(train_data,
                     test_data,
                     train_path.replace("train","train_new"),
                     test_path.replace("test","test_new"))
    return train_data, test_data

def save_new_dataset(train_data, test_data, train_path, test_path):
    train_pkl = open(train_path,"wb")
    test_pkl = open(test_path,"wb")
    pkl.dump({'packed_fp': train_data[0].numpy(),
              'values': train_data[1]},train_pkl)
    pkl.dump({'packed_fp': test_data[0].numpy(),
              'values': test_data[1]},test_pkl)
    train_pkl.close()
    test_pkl.close()

if __name__=="__main__":
    train_data, val_data = rebuild_task2(os.path.join(".","resources","MoleculeEvaluationData"))
