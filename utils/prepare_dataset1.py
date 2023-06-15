import sys
import pickle
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

data_path = Path('sources')

def get_fp(product):
    mol = Chem.MolFromSmiles(product)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    fingerprint = np.zeros(fp.GetNumBits(), dtype=bool)
    fingerprint[onbits] = 1
    return fingerprint

def get_temp(reactants, product):
    inputRec = {'_id': None, 'reactants': reactants, 'products': product}
    ans = extract_from_reaction(inputRec)
    if 'reaction_smarts' in ans.keys():
        return ans['reaction_smarts']
    else:
        return None

def generate_data():
    temp_hash = {}
    logger.remove(None)
    logger.add(Path('task1_datagen.log'))

    raw_list = ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']
    target_list = [i.replace('raw', 'task1').replace('csv', 'pkl') for i in raw_list]
    for idx in range(len(raw_list)):
        raw_data_path = raw_list[idx]
        data = pd.read_csv(data_path / raw_data_path)
        ids, classes, reactions = data.columns
        ids, classes, reactions = data[ids], data[classes], data[reactions]
        new_ids, new_classes, fingerprints, templates = [], [], [], []
        # how many templates are dropped
        drop_cnt = 0
        drop_idx = []
        with tqdm(range(len(reactions))) as tbar:
            for i in range(len(reactions)):
                reactants, products = reactions[i].split('>>')
                products = products.split('.')
                for product in products:
                    # get fingerprint
                    fp = get_fp(product)
                    # get template
                    temp = get_temp(reactants, product)
                    if temp is not None:
                        new_ids.append(ids[i])
                        new_classes.append(classes[i])
                        fingerprints.append(fp)
                        templates.append(temp)
                        temp_hash[temp] = 1
                    else:
                        drop_cnt += 1
                        drop_idx.append(i)
                        tbar.set_postfix_str(f'total drop: {drop_cnt}')
                        logger.critical(f'drop_idx: {drop_idx}')
                tbar.update()

        target_data = {
            'ids': new_ids,
            'classes': new_classes,
            'fingerprints': fingerprints,
            'templates': templates
        }
        with open(data_path / target_list[idx], 'wb') as f:
            pickle.dump(target_data, f)
        print(raw_data_path, 'done.')

    temp_hash = dict(zip(temp_hash.keys(), range(len(temp_hash))))
    with open(data_path / 'temp_hash.pkl', 'wb') as f:
        pickle.dump(temp_hash, f)

if __name__ == '__main__':
    from rdchiral.main import rdchiralRunText
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
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 0)
    plt.bar(prod_to_temp.keys(), prod_to_temp.values())
    plt.subplot(1, 2, 1)
    plt.bar(temp_to_prod.keys(), temp_to_prod.values())
    plt.savefig('./temp_vs_prod.png')