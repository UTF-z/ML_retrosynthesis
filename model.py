import torch
import torch.nn as nn
import os

act_funcs={
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "logsigmoid": nn.LogSigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax
}

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, act_func="sigmoid", last_act=None) -> None:
        super().__init__()
        assert isinstance(hid_dim,list)
        hid_layers=len(hid_dim)
        self.dims=[in_dim]
        self.dims.extend(hid_dim)
        self.dims.append(out_dim)
        self.hid_layers=hid_layers
        self.module_list=[]
        self.module_list = [
            nn.Sequential(
                nn.Linear(self.dims[j],self.dims[j+1]),
                act_funcs[act_func]()
            ) 
            for j in range(hid_layers)
        ]
        if last_act is not None:
            self.module_list.append(nn.Sequential(
                nn.Linear(self.dims[-2],self.dims[-1]),
                act_funcs[last_act]()
            ))
        else:
            self.module_list.append(nn.Linear(self.dims[-2],self.dims[-1]))
        self.mlp=nn.ModuleList(self.module_list)

    def forward(self, x):
        for j in range(self.hid_layers+1):
            x = self.mlp[j](x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self,dim,code_dim) -> None:
        super().__init__()
        self.encoder = MLP(in_dim=dim,hid_dim=[512,512],out_dim=code_dim,act_func="relu")
        self.decoder = MLP(in_dim=code_dim,hid_dim=[512,512],out_dim=dim,act_func="relu")
        self.ori_forward = self.forward

    def forward(self,x):
        return self.decoder(self.encoder(x))
    
    def encode(self,x):
        return self.encoder(x)
    
    def embed(self):
        self.forward=self.encode

    def retrain(self):
        self.forward=self.ori_forward