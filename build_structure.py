import os
import torch
import pickle as pkl

os.makedirs("resources",exist_ok=True)
os.makedirs("checkpoint",exist_ok=True)
os.system('gzip -d resources/MoleculeEvaluationData/test.pkl.gz')
os.system('gzip -d resources/MoleculeEvaluationData/train.pkl.gz')
os.system('gzip -d "resources/Multi-Step task/starting_mols.pkl.gz"')