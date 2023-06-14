import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_best(model,path):
    torch.save({"model":model.state_dict()},path)

def load_best(path):
    return torch.load(path)