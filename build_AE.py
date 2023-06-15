import torch
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

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
        train_data = (np.unpackbits(train_data['packed_fp']).reshape(train_size,-1),train_data['values']) # (uint8, fp32)
        test_data = (np.unpackbits(test_data['packed_fp']).reshape(test_size,-1),test_data['values']) # (uint8, fp32)
    else:
        train_data = (train_data['packed_fp'],train_data['values']) # (uint8, fp32)
        test_data = (test_data['packed_fp'],test_data['values']) # (uint8, fp32)
    return train_data, test_data

def read_data_task1(root_dir=os.path.join(".","task1_data"),dtype=str) -> np.ndarray:
    train_path = os.path.join(root_dir,"task1_train.pkl")
    train_pkl = open(train_path,"rb")
    train_data = pkl.load(train_pkl)
    fingerprints = train_data['fingerprints']
    fp_list = []
    for fp in fingerprints:
        fp_list.append(fp.tolist())
    fingerprints = np.array(np.array(fp_list,dtype=np.uint8),dtype=dtype)
    return fingerprints

def build_VAE_data(root_dir=os.path.join(".","resources")):
    train_data,_ = build_data_task2(os.path.join(root_dir,"MoleculeEvaluationData"))
    task1 = read_data_task1(os.path.join(root_dir),dtype=np.uint8).repeat(10,axis=0)
    task2 = train_data[0]
    data_all = np.concatenate([task1,task2],axis=0)
    print(data_all.shape)
    pack = {
        'ids': None,
        'classes': None,
        'fingerprints': data_all,
        'templates': None
    }
    return pack

def build_data_task3(root_dir=os.path.join(".","MoleculeEvaluationData"),unpack=True):
    train_path = os.path.join(root_dir,"train_mol_fp_value_step.pt")
    train_pkl = open(train_path,"rb")
    test_path = os.path.join(root_dir,"val_mol_fp_value_step.pt")
    test_pkl = open(test_path,"rb")
    train_data = torch.load(train_pkl)
    test_data = torch.load(test_pkl)
    # train_size = train_data['packed_fp'].shape[0]
    # test_size = test_data['packed_fp'].shape[0]
    # if unpack:
    #     train_data = (np.unpackbits(train_data['packed_fp']).reshape(train_size,-1),) # (uint8, fp32)
    #     test_data = (np.unpackbits(test_data['packed_fp']).reshape(test_size,-1),test_data['values']) # (uint8, fp32)
    # else:
    #     train_data = (train_data['packed_fp'],train_data['values']) # (uint8, fp32)
    #     test_data = (test_data['packed_fp'],test_data['values']) # (uint8, fp32)
    return train_data, test_data

if __name__ == "__main__":
    # train_data,_ = build_data_task2()
    # task1 = read_data_task1()
    # task2 = train_data[0]
    # print(task1)
    # print(task1.shape)
    # print(task2)
    # print(task2.shape)
    # task1_dict = {}
    # for i in tqdm(range(task1.shape[0])):
    #     string = "".join(task1[i].tolist())
    #     task1_dict[string]=1
    # cnt = 0
    # tbar = tqdm(range(task2.shape[0]))
    # for j in range(task2.shape[0]):
    #     string = "".join(np.array(task2[j],dtype=str).tolist())
    #     cnt += 1 if string in task1_dict else 0
    #     tbar.set_postfix_str(f"cnt: {cnt}")
    #     tbar.update()
    # print(cnt)
    if True:
        VAE_data = build_VAE_data()
        os.makedirs("./resources/pretrain/",exist_ok=True)
        VAE_file = open("./resources/pretrain/AE_data.pkl","wb")
        pkl.dump(VAE_data,VAE_file)