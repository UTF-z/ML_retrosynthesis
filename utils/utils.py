import torch
import torch.nn as nn
import sys
sys.path.append('.')
import pickle
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRunText
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit.Chem.AllChem import DataStructs
from dataset.single_step import SingleStep
import rdkit

data_path = Path('resources')

def get_fp(product):
    mol = Chem.MolFromSmiles(product)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    fingerprint = np.zeros(fp.GetNumBits(), dtype=bool)
    fingerprint[onbits] = 1
    return fingerprint, fp

def get_temp(reactants, product):
    inputRec = {'_id': None, 'reactants': reactants, 'products': product}
    ans = extract_from_reaction(inputRec)
    if 'reaction_smarts' in ans.keys():
        return ans['reaction_smarts']
    else:
        return None

def get_similarity(fp1: np.ndarray, fp2: np.ndarray):
    ifp1 = DataStructs.cDataStructs.ExplicitBitVect(2048)
    fp1 = np.nonzero(fp1)[0].tolist()
    ifp1.SetBitsFromList(fp1)
    ifp2 = DataStructs.cDataStructs.ExplicitBitVect(2048)
    fp2 = np.nonzero(fp2)[0].tolist()
    ifp2.SetBitsFromList(fp2)
    sim = DataStructs.DiceSimilarity(ifp1, ifp2)
    return sim


def generate_data():
    from loguru import logger
    temp_hash = {}
    logger.remove(None)
    logger.add(Path('task1_datagen.log'))

    raw_list = ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']
    target_list = [i.replace('raw', 'task1').replace('csv', 'pkl') for i in raw_list]
    train_temp_record = {}
    for idx in range(len(raw_list)):
        raw_data_path = raw_list[idx]
        data = pd.read_csv(data_path / raw_data_path)
        ids, classes, reactions = data.columns
        ids, classes, reactions = data[ids], data[classes], data[reactions]
        new_ids, new_classes, smiles, fingerprints, templates = [], [], [], [], []
        # how many templates are dropped
        drop_cnt = 0
        drop_idx = []
        total_cnt = 0
        new_cnt = 0
        fail_cnt = 0
        debug_total_cnt = 0
        with tqdm(range(len(reactions))) as tbar:
            for i in range(len(reactions)):
                reactants, products = reactions[i].split('>>')
                products = products.split('.')
                for product in products:
                    # get fingerprint
                    fp, raw_fp = get_fp(product)
                    # get template
                    temp = get_temp(reactants, product)
                    if temp is not None:
                        new_ids.append(ids[i])
                        new_classes.append(classes[i])
                        fingerprints.append(fp)
                        templates.append(temp)
                        smiles.append(product)
                        reactants = get_reactants(smile=product, temp=temp)
                        if reactants is None:
                            fail_cnt += 1
                        debug_total_cnt += 1
                        _exist = train_temp_record.get(temp, 0)
                        if _exist == 0:
                            new_cnt += 1
                        temp_hash[temp] = 1
                        total_cnt += 1
                    else:
                        drop_cnt += 1
                        drop_idx.append(i)
                        tbar.set_postfix_str(f'total drop: {drop_cnt}')
                        logger.critical(f'drop_idx: {drop_idx}')
                tbar.set_postfix_str(f'fail_ratio: {fail_cnt/debug_total_cnt}')
                tbar.update()
        if raw_data_path == 'raw_train.csv':
            train_temp_record.update(temp_hash)
        target_data = {
            'ids': new_ids,
            'classes': new_classes,
            'fingerprints': fingerprints,
            'smiles': smiles,
            'templates': templates
        }
        logger.critical(f'{raw_data_path}: total temps: {total_cnt}, new temps: {new_cnt}')
        with open(data_path / target_list[idx], 'wb') as f:
            pickle.dump(target_data, f)
        print(raw_data_path, 'done.')
    temp_hash = dict(zip(temp_hash.keys(), range(len(temp_hash))))
    with open(data_path / 'temp_hash.pkl', 'wb') as f:
        pickle.dump(temp_hash, f)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_best(model,path):
    torch.save({"model":model.state_dict()},path)

def load_best(path):
    return torch.load(path)['model']

def vis_prod_vs_temp():
    data = pd.read_csv('resources/raw_test.csv')
    a, b, c = data.columns
    reactions = data[c]
    temps = []
    prods = []
    for i in tqdm(range(len(reactions))):
        r = reactions[i]
        rt, pt = r.split('>>')
        temp = get_temp(rt, pt)
        temps.append(temp)
        prods.append(pt)
    prod_to_temp = {}
    temp_to_prod = {}
    with tqdm(range(len(prods))) as tbar:
        for i in range(len(prods)):
            pt = prods[i]
            cnt = 0
            for j in range(len(temps)):
                temp = temps[j]
                try:
                    out = rdchiralRunText(temp, pt)
                except Exception as e:
                    pass
                if len(out) > 0:
                    cnt += 1
                    temp_to_prod[temp] = temp_to_prod.get(temp, 0) + 1
            tbar.update()
            tbar.set_postfix_str(f'cnt: {cnt}')
            print(cnt)
            prod_to_temp[pt] = cnt
    with open('prod_to_temp.pkl', 'wb') as f:
        pickle.dump(prod_to_temp, f)
    with open('temp_to_prod.pkl', 'wb') as f:
        pickle.dump(temp_to_prod, f)
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.bar(prod_to_temp.keys(), prod_to_temp.values())
    plt.subplot(1, 2, 2)
    plt.bar(temp_to_prod.keys(), temp_to_prod.values())
    plt.savefig('./temp_vs_prod.png')

def fp_ndarray_to_fp(fp: np.ndarray):
    ifp1 = DataStructs.cDataStructs.ExplicitBitVect(2048)
    fp1 = np.nonzero(fp)[0].tolist()
    ifp1.SetBitsFromList(fp1)
    return ifp1

def is_compatible(smile, template):
    try:
        out = rdchiralRunText(template, smile)
    except Exception as e:
        return False
    if len(out) > 0:
        return True
    else:
        return False

def dataspliter(dataset: torch.Tensor,
                labels: torch.Tensor,
                train_ratio: float,
                device='cpu'):
    """
    Input:
        dataset (Tensor): total dataset [B,...]
        labels (Tensor): total labels [B,...]
    Output:
        train_set,train_labels,val_set,val_labels
    """
    assert dataset.shape[0] == labels.shape[0]
    total_size = dataset.shape[0]
    data_shape = tuple(torch.cat([torch.tensor([-1]),torch.ones(len(list(dataset.shape))-1)],dim=0).int().numpy())
    labels_shape = tuple(torch.cat([torch.tensor([-1]),torch.ones(len(list(labels.shape))-1)],dim=0).int().numpy())
    data_repeat = list(dataset.shape)
    data_repeat[0] = 1
    data_repeat = tuple(data_repeat)
    labels_repeat = list(labels.shape)
    labels_repeat[0] = 1
    labels_repeat = tuple(labels_repeat)
    rand_id = torch.randperm(total_size)
    data_id = rand_id.reshape(data_shape).repeat(data_repeat).to(device=device)
    labels_id = rand_id.reshape(labels_shape).repeat(labels_repeat).to(device=device)
    rand_set = torch.gather(dataset,dim=0,index=data_id)
    rand_labels = torch.gather(labels,dim=0,index=labels_id)
    train_set = rand_set[:int(total_size*train_ratio)]
    train_labels = rand_labels[:int(total_size*train_ratio)]
    val_set = rand_set[int(total_size*train_ratio):]
    val_labels = rand_labels[int(total_size*train_ratio):]
    return (train_set,train_labels),(val_set,val_labels)


def get_reactants(smile, temp):
    try:
        out = rdchiralRunText(temp, smile)
        assert len(out) > 0, f'out: {out}, temp: {temp}, smile: {smile}'
        out = out[0].split('.')
        return out
    except Exception as e:
        return None


if __name__ == '__main__':
    generate_data()